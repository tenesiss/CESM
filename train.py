#!/usr/bin/env python3
"""
Training and shared model definitions for the causal Unnoba video-to-text prototype.

Data format (JSONL, one example per line):
    {"video": "clips/hello.mp4", "text": "hello",
     "windows": [[0,3],[4,6],[7,9],[10,12],[13,16]]}

By default the tokenizer is character-level, so windows[i] is the inclusive
[start_frame, end_frame] interval in which text[i] is uttered.  Passing
--pretrained-lm switches both tokenization and the causal text encoder to a
Hugging Face causal LM.  Character windows are then merged using the fast
tokenizer's offset mapping so every LM token retains causal video alignment.
Token-query variants add an EOS target at the final video frame. The frame-token
variant supervises frames inside real token windows and does not train EOS.

Core architecture:
  video: fixed mouth ROI -> causal Conv3D -> per-frame residual encoder
         -> sinusoidal positions -> causal Conformer-like stack
  text:  teacher token prefix -> causal text encoder
  fusion: projected video/text states -> Mixed Block -> token/confidence heads

Token-query variants use text->video attention for tokens and video->text
attention for confidence. The frame-token variant uses video->text attention
for both heads and has no text->video attention branch.

Training has two stages:
  1. token or frame cross entropy, with configurable projection and attention
     regularizers. By default, only projection norm penalties have nonzero
     weights; geometry, monotonic attention, aligned-window attention and
     tcross norm penalties are disabled.
  2. confidence training with the token path frozen, following the document's
     window/timestep equations directly. For every aligned token window, the
     frozen token predictor is evaluated at each frame using only the video
     prefix available at that frame. They are reduced window-by-window using the
     logical Y[B,W,T,V] layout without materializing that dense tensor; frame
     confidence logits are packed with the same (W,T) layout.
     The confidence target is s = 1 - S, where S mixes normalized token entropy
     U and future-distribution instability Q (Jensen-Shannon divergence with the
     document's log-sum-exp aggregation).

Optionally, --pretrain-visual-encoder first trains the video encoder on an
unannotated --pretrain-manifest using augmented-view MSE, variance and covariance
regularization, and optional temporal MSE.

The supplied frame windows are also used as causal supervision: target token t
may only cross-attend to frames up through the end of its own window. The
video->text path receives the dual causal mask: a frame can only attend to text
states that would already exist by that frame in streaming inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

from flash_mono import attention_statistics_from_qk
from learning_curves import (
    LearningCurveLogger, accuracy_decode_args, add_learning_curve_arguments, token_edit_distance,
)
from streaming import decode_stream


# -----------------------------------------------------------------------------
# Repro / config
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_activation(x: torch.Tensor) -> torch.Tensor:
    """Cast activations to the AMP dtype; preserve FP64 and non-AMP inputs."""
    if x.dtype != torch.float64 and torch.is_autocast_enabled(x.device.type):
        return x.to(dtype=torch.get_autocast_dtype(x.device.type))
    return x


class AutocastGroupNorm(nn.GroupNorm):
    """Use FP32 math and low-precision output under AMP, except for FP64 inputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float64 or not torch.is_autocast_enabled(x.device.type):
            return super().forward(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = F.group_norm(
                x.float(), self.num_groups,
                self.weight.float() if self.weight is not None else None,
                self.bias.float() if self.bias is not None else None, self.eps,
            )
        return autocast_activation(y)


class AutocastLayerNorm(nn.LayerNorm):
    """Use FP32 math under AMP except for FP64 inputs; keep LayerNorm state keys."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float64 or not torch.is_autocast_enabled(x.device.type):
            return super().forward(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = F.layer_norm(
                x.float(), self.normalized_shape,
                self.weight.float() if self.weight is not None else None,
                self.bias.float() if self.bias is not None else None, self.eps,
            )
        return autocast_activation(y)


@dataclass
class ModelConfig:
    vocab_size: int
    mouth_size: int = 96
    conv3d_channels: int = 32
    d_video: int = 256
    d_text: int = 256
    d_fusion: int = 256
    heads: int = 4
    video_layers: int = 4
    text_layers: int = 2
    ff_mult: int = 4
    conv_kernel: int = 7
    dropout: float = 0.1
    text_encoder_type: str = "learned"
    pretrained_lm_name_or_path: Optional[str] = None
    pretrained_lm_config: Optional[Dict[str, Any]] = None
    freeze_text_encoder: bool = False
    pretrained_lm_local_files_only: bool = False
    text_fusion: str = "residual"
    text_residual_weight: float = 1.0
    use_video_projection: bool = True
    frame_vcross_weight: float = 1.0


# -----------------------------------------------------------------------------
# Character and pretrained tokenizers
# -----------------------------------------------------------------------------


class CharTokenizer:
    """Character tokenizer so the frame-window mapping stays explicit."""

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, itos: Sequence[str]):
        self.itos = list(itos)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        for tok in (self.PAD, self.BOS, self.EOS, self.UNK):
            if tok not in self.stoi:
                raise ValueError(f"Tokenizer is missing special token {tok!r}")

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharTokenizer":
        chars = sorted({ch for text in texts for ch in text})
        return cls([cls.PAD, cls.BOS, cls.EOS, cls.UNK] + chars)

    @property
    def pad_id(self) -> int:
        return self.stoi[self.PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.UNK]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        return self.encode(text), [(i, i + 1) for i in range(len(text))]

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for idx in ids:
            tok = self.itos[int(idx)]
            if tok == self.EOS:
                break
            if tok in (self.PAD, self.BOS):
                continue
            out.append("�" if tok == self.UNK else tok)
        return "".join(out)

    def state_dict(self) -> Dict[str, Any]:
        return {"type": "char", "itos": self.itos}

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "CharTokenizer":
        return cls(state["itos"])


def _require_huggingface():
    try:
        from tokenizers import Tokenizer
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "Using --pretrained-lm requires the optional Hugging Face dependencies. "
            "Install them with: pip install transformers tokenizers"
        ) from exc
    return Tokenizer, AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


class HuggingFaceTokenizer:
    """Small, checkpoint-serializable adapter around a fast HF tokenizer."""

    def __init__(self, tokenizer, name_or_path: str):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                f"{name_or_path!r} does not provide a fast tokenizer; CESM needs "
                "offset mappings to convert character frame windows to LM-token windows"
            )
        self.tokenizer = tokenizer
        self.name_or_path = name_or_path
        self._ensure_distinct_special_tokens()

    def _new_special_token(self, base: str) -> str:
        vocab = self.tokenizer.get_vocab()
        candidate = base
        suffix = 1
        while candidate in vocab:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    def _ensure_distinct_special_tokens(self) -> None:
        tok = self.tokenizer
        if tok.eos_token_id is None:
            tok.add_special_tokens({"eos_token": self._new_special_token("<|cesm_eos|>")})
        if tok.bos_token_id is None:
            # Reuse EOS as initial context when BOS is absent, avoiding a
            # separate BOS embedding.
            tok.bos_token = tok.eos_token
        if tok.pad_token_id is None or tok.pad_token_id in (tok.bos_token_id, tok.eos_token_id):
            tok.add_special_tokens({"pad_token": self._new_special_token("<|cesm_pad|>")})
        if None in (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id):
            raise ValueError("The pretrained tokenizer must define PAD, BOS and EOS token ids")
        if tok.pad_token_id in (tok.bos_token_id, tok.eos_token_id):
            raise ValueError("The pretrained tokenizer's PAD id must be distinct from BOS/EOS")

    @classmethod
    def from_pretrained(cls, name_or_path: str, *, local_files_only: bool = False):
        _, _, _, AutoTokenizer, _ = _require_huggingface()
        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path,
            use_fast=True,
            local_files_only=local_files_only,
        )
        return cls(tokenizer, name_or_path)

    @property
    def pad_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    @property
    def bos_id(self) -> int:
        return int(self.tokenizer.bos_token_id)

    @property
    def eos_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def unk_id(self) -> int:
        value = self.tokenizer.unk_token_id
        return int(value) if value is not None else self.eos_id

    @property
    def vocab_size(self) -> int:
        # len(tokenizer), unlike tokenizer.vocab_size, includes CESM special
        # tokens added to models such as GPT-2.
        return len(self.tokenizer)

    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_offsets_mapping=True,
        )
        ids = [int(x) for x in encoded["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
        if len(ids) != len(offsets):
            raise RuntimeError("Pretrained tokenizer returned inconsistent ids and offsets")
        return ids, offsets

    def decode(self, ids: Iterable[int]) -> str:
        return self.tokenizer.decode(
            [int(x) for x in ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": "huggingface_fast",
            "name_or_path": self.name_or_path,
            "backend_json": self.tokenizer.backend_tokenizer.to_str(),
            "special_tokens": {
                "pad_token": self.tokenizer.pad_token,
                "bos_token": self.tokenizer.bos_token,
                "eos_token": self.tokenizer.eos_token,
                "unk_token": self.tokenizer.unk_token,
                "sep_token": self.tokenizer.sep_token,
                "cls_token": self.tokenizer.cls_token,
                "mask_token": self.tokenizer.mask_token,
            },
            "model_max_length": int(self.tokenizer.model_max_length),
            "padding_side": self.tokenizer.padding_side,
            "truncation_side": self.tokenizer.truncation_side,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]):
        Tokenizer, _, _, _, PreTrainedTokenizerFast = _require_huggingface()
        backend = Tokenizer.from_str(state["backend_json"])
        kwargs = {k: v for k, v in state.get("special_tokens", {}).items() if v is not None}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            model_max_length=state.get("model_max_length", int(1e30)),
            padding_side=state.get("padding_side", "right"),
            truncation_side=state.get("truncation_side", "right"),
            **kwargs,
        )
        return cls(tokenizer, state.get("name_or_path", "checkpoint-tokenizer"))


def tokenizer_from_state_dict(state: Dict[str, Any]):
    tokenizer_type = state.get("type", "char")  # Older checkpoints had no type field.
    if tokenizer_type == "char":
        return CharTokenizer.from_state_dict(state)
    if tokenizer_type == "huggingface_fast":
        return HuggingFaceTokenizer.from_state_dict(state)
    raise ValueError(f"Unsupported checkpoint tokenizer type: {tokenizer_type!r}")


# -----------------------------------------------------------------------------
# Video preprocessing: fixed mouth ROI with a causal-friendly detector fallback
# -----------------------------------------------------------------------------


class FixedMouthCropper:
    """
    Prototype replacement for the document's landmark/smoothing/fixed-mouth-ROI
    front end.

    The first frame fixes the rectangle for the entire sequence, using its
    largest detected face or a lower-center fallback. Detection is not retried
    on later frames, so crop selection never reads future frames.
    """

    def __init__(self, size: int = 96, use_face_detector: bool = True):
        self.size = size
        self.use_face_detector = use_face_detector
        self.rect: Optional[Tuple[int, int, int, int]] = None
        self.detector = None
        if use_face_detector:
            if not hasattr(cv2, "CascadeClassifier"):
                raise RuntimeError(
                    "The Haar face detector requires OpenCV 4.x. Install the project's "
                    "pinned dependencies with '.venv/bin/python -m pip install -r "
                    "requirements.txt', or pass --no-face-detector to use the "
                    "deterministic lower-center crop."
                )
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            self.detector = cv2.CascadeClassifier(cascade_path)

    def _fallback_rect(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        x0, x1 = int(0.20 * w), int(0.80 * w)
        y0, y1 = int(0.45 * h), int(0.95 * h)
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def _detect_rect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if self.detector is None or self.detector.empty():
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return None
        # Prefer the largest detected face.
        x, y, w, h = max(faces, key=lambda r: int(r[2]) * int(r[3]))
        # Lower-central region of the face as a simple mouth proxy.
        mx0 = int(x + 0.12 * w)
        mx1 = int(x + 0.88 * w)
        my0 = int(y + 0.52 * h)
        my1 = int(y + 0.98 * h)
        return mx0, my0, max(1, mx1 - mx0), max(1, my1 - my0)

    @staticmethod
    def _clip_rect(rect: Tuple[int, int, int, int], frame: np.ndarray) -> Tuple[int, int, int, int]:
        x, y, w, h = rect
        H, W = frame.shape[:2]
        x0 = max(0, min(W - 1, x))
        y0 = max(0, min(H - 1, y))
        x1 = max(x0 + 1, min(W, x + w))
        y1 = max(y0 + 1, min(H, y + h))
        return x0, y0, x1 - x0, y1 - y0

    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        if self.rect is None:
            self.rect = self._detect_rect(frame) or self._fallback_rect(frame)
        x, y, w, h = self._clip_rect(self.rect, frame)
        crop = frame[y : y + h, x : x + w]
        crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 127.5 - 1.0
        return tensor


def read_video_mouth_tensor(
    path: str,
    mouth_size: int,
    use_face_detector: bool = True,
    *,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    video_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, float]:
    """Read a bounded section while keeping the original video's fixed ROI."""
    if start_frame < 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("start_frame must be >= 0 and max_frames must be > 0")
    cap = cv2.VideoCapture(path)
    frames: List[torch.Tensor] = []
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        cropper = FixedMouthCropper(mouth_size, use_face_detector=use_face_detector)
        if start_frame:
            ok, first_frame = cap.read()
            if not ok:
                raise RuntimeError(f"Video has no decodable frames: {path}")
            cropper(first_frame)  # Every section uses the first frame's ROI.
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
                # Some backends cannot seek. Skip without retaining tensors.
                cap.release()
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError(f"Could not reopen video: {path}")
                for _ in range(start_frame):
                    if not cap.grab():
                        raise RuntimeError(f"Could not reach frame {start_frame}: {path}")
        while max_frames is None or len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            # Normalize pixels in FP32, but retain only the requested storage
            # dtype. AMP callers pass it explicitly for DataLoader workers.
            frames.append(cropper(frame).to(dtype=video_dtype))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Video has no decodable frames at frame {start_frame}: {path}")
    return torch.stack(frames, dim=0), fps  # [F,3,H,W]


def count_video_frames(path: str) -> int:
    """Count successful frame grabs instead of using container frame-count metadata."""
    cap = cv2.VideoCapture(path)
    count = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        while cap.grab():
            count += 1
    finally:
        cap.release()
    if count == 0:
        raise RuntimeError(f"Video has no decodable frames: {path}")
    return count


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


def load_manifest(path: str, *, video_only: bool = False) -> List[dict]:
    """Resolve video paths; self-supervised manifests need no annotations."""
    base = Path(path).resolve().parent
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in (("video",) if video_only else ("video", "text", "windows")):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: missing key {key!r}")
            vp = Path(row["video"])
            if not vp.is_absolute():
                vp = base / vp
            row["video"] = str(vp)
            rows.append(row)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


class PretrainVideoDataset(Dataset):
    """Unannotated mouth videos, with the same bounded decoding as training."""

    def __init__(self, rows, mouth_size, *, use_face_detector=True,
                 max_frames=None, video_dtype=torch.float32):
        if max_frames is not None and max_frames <= 0:
            raise ValueError("--max-frames must be > 0")
        self.rows = list(rows)
        self.mouth_size = mouth_size
        self.use_face_detector = use_face_detector
        self.video_dtype = video_dtype
        self.sections = []
        frame_counts = {}
        for row_idx, row in enumerate(self.rows):
            if max_frames is None:
                self.sections.append((row_idx, 0, None))
                continue
            path = row["video"]
            if path not in frame_counts:
                frame_counts[path] = count_video_frames(path)
            count = frame_counts[path]
            for start in range(0, count, max_frames):
                self.sections.append((row_idx, start, min(max_frames, count - start)))

    def __len__(self):
        return len(self.sections)

    def __getitem__(self, idx):
        row_idx, start, count = self.sections[idx]
        path = self.rows[row_idx]["video"]
        video, _ = read_video_mouth_tensor(
            path, self.mouth_size, use_face_detector=self.use_face_detector,
            start_frame=start, max_frames=count, video_dtype=self.video_dtype,
        )
        if count is not None and len(video) != count:
            raise RuntimeError(f"{path}: could not decode indexed section [{start}, {start + count})")
        return {"video": video, "path": path}


def collate_pretrain_videos(batch: Sequence[dict]) -> dict:
    lengths = torch.tensor([len(item["video"]) for item in batch], dtype=torch.long)
    sample = batch[0]["video"]
    video = sample.new_zeros(len(batch), int(lengths.max()), *sample.shape[1:])
    for i, item in enumerate(batch):
        video[i, :len(item["video"])] = item["video"]
    return {"video": video, "video_lengths": lengths}


def video_identity_keys(rows: Sequence[dict]) -> set:
    """Match local videos and, when recorded, their original source uploads."""
    keys = {("path", str(Path(row["video"]).resolve())) for row in rows}
    keys.update(("source", row["source_key"]) for row in rows if row.get("source_key"))
    return keys


def validate_validation_split(training_rows, validation_rows) -> None:
    if video_identity_keys(training_rows) & video_identity_keys(validation_rows):
        raise ValueError("Training and validation manifests overlap: use separate videos/source uploads")


def repair_quantized_empty_windows(
    windows: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], int]:
    """Give repairable zero-frame intervals one frame from an adjacent interval.

    Timestamp-to-frame rounding can turn a valid sub-frame token interval into
    ``[start, start - 1]``.  Repair only that exact one-frame collapse, and only
    when an immediately adjacent, contiguous interval has more than one frame.
    Callers validate any remaining malformed alignments.
    """
    repaired = list(windows)
    repair_count = 0
    for i, (start, end) in enumerate(repaired):
        if end != start - 1:
            continue

        # Prefer the following interval so the repaired token remains at its
        # rounded start time.
        if i + 1 < len(repaired):
            next_start, next_end = repaired[i + 1]
            if next_start == start and next_end > next_start:
                repaired[i] = (start, start)
                repaired[i + 1] = (next_start + 1, next_end)
                repair_count += 1
                continue

        # If the following interval cannot donate, try the preceding interval's
        # final frame instead.
        if i > 0:
            prev_start, prev_end = repaired[i - 1]
            if prev_end == end and prev_end > prev_start:
                repaired[i - 1] = (prev_start, prev_end - 1)
                repaired[i] = (end, end)
                repair_count += 1

    return repaired, repair_count


class VideoTextWindowDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict],
        tokenizer,
        mouth_size: int,
        use_face_detector: bool = True,
        max_frames: Optional[int] = None,
        video_dtype: torch.dtype = torch.float32,
        window_phasing: float = 0.0,
    ):
        if max_frames is not None and max_frames <= 0:
            raise ValueError("--max-frames must be > 0")
        if not math.isfinite(window_phasing) or not 0 <= window_phasing <= 1:
            raise ValueError("--window-phasing must be finite and in [0,1]")
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.mouth_size = mouth_size
        self.use_face_detector = use_face_detector
        self.video_dtype = video_dtype
        self.window_phasing = window_phasing
        self.sections: Optional[List[Tuple[int, int, int, int]]] = None
        if max_frames is not None:
            # Index before batching so batch_size counts sections, not videos.
            # Store section bounds and frame counts; load tensors on demand.
            self.sections = []
            frame_counts: Dict[str, int] = {}
            for row_idx, row in enumerate(self.rows):
                path = row["video"]
                if path not in frame_counts:
                    frame_counts[path] = count_video_frames(path)
                count = frame_counts[path]
                for start in range(0, count, max_frames):
                    self.sections.append((row_idx, start, min(start + max_frames, count), count))

    def __len__(self) -> int:
        return len(self.rows) if self.sections is None else len(self.sections)

    def __getitem__(self, idx: int) -> dict:
        start_frame = 0
        stop_frame = full_frame_count = None
        if self.sections is not None:
            idx, start_frame, stop_frame, full_frame_count = self.sections[idx]
        row = self.rows[idx]
        text = row["text"]
        windows = row["windows"]
        window_unit = row.get("window_unit")
        if window_unit not in (None, "character", "token"):
            raise ValueError(
                f"{row['video']}: window_unit must be 'character' or 'token', got {window_unit!r}"
            )
        mouth, fps = read_video_mouth_tensor(
            row["video"], self.mouth_size, use_face_detector=self.use_face_detector,
            start_frame=start_frame,
            max_frames=None if stop_frame is None else stop_frame - start_frame,
            video_dtype=self.video_dtype,
        )
        if stop_frame is not None and mouth.shape[0] != stop_frame - start_frame:
            raise RuntimeError(
                f"{row['video']}: could not decode indexed section [{start_frame}, {stop_frame})"
            )
        Fv = mouth.shape[0] if full_frame_count is None else full_frame_count
        parsed_windows: List[Tuple[int, int]] = []
        for i, win in enumerate(windows):
            if not isinstance(win, (list, tuple)) or len(win) != 2:
                raise ValueError(f"{row['video']}: windows[{i}] must be [start_frame,end_frame]")
            start, end = int(win[0]), int(win[1])
            parsed_windows.append((start, end))

        clean_windows, repaired_count = repair_quantized_empty_windows(parsed_windows)
        if repaired_count:
            warnings.warn(
                f"{row['video']}: repaired {repaired_count} zero-frame window(s) "
                "caused by timestamp-to-frame quantization",
                stacklevel=2,
            )

        last_end = -1
        for start, end in clean_windows:
            if start < 0 or end < start or end >= Fv:
                raise ValueError(
                    f"{row['video']}: invalid window [{start}, {end}]; video has {Fv} frames"
                )
            if start <= last_end:
                raise ValueError(f"{row['video']}: windows must be ordered and non-overlapping")
            last_end = end

        token_ids, offsets = self.tokenizer.encode_with_offsets(text)
        use_character_windows = window_unit == "character" or (
            window_unit is None and len(clean_windows) == len(text)
        )
        use_token_windows = window_unit == "token" or (
            window_unit is None
            and len(clean_windows) != len(text)
            and len(clean_windows) == len(token_ids)
        )
        if use_character_windows:
            if len(clean_windows) != len(text):
                raise ValueError(
                    f"{row['video']}: window_unit='character' requires {len(text)} windows, "
                    f"got {len(clean_windows)}"
                )
            token_windows: List[Tuple[int, int]] = []
            for token_idx, (char_start, char_end) in enumerate(offsets):
                if not (0 <= char_start < char_end <= len(text)):
                    raise ValueError(
                        f"{row['video']}: tokenizer returned unusable offset "
                        f"{(char_start, char_end)} for token {token_idx}; use a fast causal-LM "
                        "tokenizer whose offsets refer to the original text"
                    )
                covered = clean_windows[char_start:char_end]
                token_windows.append((covered[0][0], covered[-1][1]))

            # Byte-level tokenizers sometimes represent one Unicode character
            # with multiple ids that all report the same character offset. A
            # shared frame window would make per-token confidence targets
            # ambiguous, so divide that character's frames deterministically.
            split_windows: List[Tuple[int, int]] = []
            i = 0
            while i < len(token_windows):
                j = i + 1
                while j < len(token_windows) and offsets[j] == offsets[i]:
                    j += 1
                count = j - i
                start, end = token_windows[i]
                frame_count = end - start + 1
                if count > frame_count:
                    raise ValueError(
                        f"{row['video']}: {count} LM tokens map to character offset {offsets[i]}, "
                        f"but its window has only {frame_count} frames; provide token-aligned "
                        "windows (one non-overlapping window per encoded token)"
                    )
                base, extra = divmod(frame_count, count)
                cursor = start
                for part in range(count):
                    length = base + (1 if part < extra else 0)
                    split_windows.append((cursor, cursor + length - 1))
                    cursor += length
                i = j
            token_windows = split_windows
        elif use_token_windows:
            if len(clean_windows) != len(token_ids):
                raise ValueError(
                    f"{row['video']}: window_unit='token' requires {len(token_ids)} windows, "
                    f"got {len(clean_windows)}"
                )
            token_windows = clean_windows
        else:
            raise ValueError(
                f"{row['video']}: expected one window per source character ({len(text)}) "
                f"or per encoded LM token ({len(token_ids)}), got {len(clean_windows)}"
            )

        last_token_end = -1
        for token_idx, (start, end) in enumerate(token_windows):
            if start <= last_token_end:
                raise ValueError(
                    f"{row['video']}: tokenizer offsets produce overlapping frame windows "
                    f"at token {token_idx}; provide one non-overlapping window per encoded token"
                )
            last_token_end = end

        # Keep the original causal text prefix. A memory boundary must not
        # turn a mid-sentence token into a new BOS prediction.
        prefix_count = 0
        all_token_ids = token_ids
        all_token_windows = token_windows
        is_video_end = stop_frame is None or stop_frame == full_frame_count
        if stop_frame is not None:
            # Assign each token exactly once, to the section where it finishes.
            # Tokenize the complete text first to preserve pretrained token IDs.
            selected = [i for i, (_, end) in enumerate(token_windows)
                        if start_frame <= end < stop_frame]
            prefix_count = sum(end < start_frame for _, end in token_windows)
            token_ids = [token_ids[i] for i in selected]
            token_windows = [
                (max(start_frame, token_windows[i][0]) - start_frame,
                 token_windows[i][1] - start_frame)
                for i in selected
            ]
            if start_frame != 0 or stop_frame != full_frame_count:
                text = self.tokenizer.decode(token_ids)
        Fv = mouth.shape[0]

        # Encode BOS from the true video start and all committed text through
        # this section. Only queries beginning at prefix_count incur token
        # loss. Causal text attention keeps target i out of its own query i.
        # Retain the last token's state for confidence after its commit, even
        # when this section has no EOS query.
        teacher_end = prefix_count + len(token_ids)
        teacher = [self.tokenizer.bos_id] + all_token_ids[:teacher_end]
        targets = token_ids + ([self.tokenizer.eos_id] if is_video_end else [])
        target_windows = token_windows + ([(Fv - 1, Fv - 1)] if is_video_end else [])

        # Query t predicts target t. It can see video through the end of that
        # target's aligned window. EOS exists only at the true video end.
        token_video_end = [end for _, end in target_windows]

        # Teacher state 0 (BOS) exists before frame 0. State t>0 contains
        # target token t-1 and is created only after processing the final frame
        # of that token's window, so the video path may use it starting at the
        # following frame.
        # Availability is relative to section start, so prior states can have
        # negative times.
        text_available_at = [-start_frame] + [
            end + 1 - start_frame for _, end in all_token_windows[:teacher_end]
        ]

        # Frame supervision intersects every original window, including tokens
        # that finish in a later section. Gaps have no known true class.
        frame_targets = torch.full((Fv,), self.tokenizer.pad_id, dtype=torch.long)
        frame_previous_targets = torch.full_like(frame_targets, self.tokenizer.pad_id)
        frame_windows = []
        for token_idx, (token_id, (window_start, window_end)) in enumerate(zip(all_token_ids, all_token_windows)):
            start = max(window_start, start_frame) - start_frame
            end = min(window_end, start_frame + Fv - 1) - start_frame
            if start <= end:
                frame_targets[start:end + 1] = token_id
                frame_windows.append((start, end))
                # Compute the inclusive phase endpoint on the original window,
                # before section clipping. Never restart phasing at a cut.
                phase_last = math.floor(self.window_phasing * (window_end - window_start + 1)) - 1
                if token_idx > 0 and phase_last >= 0:
                    phase_end = min(end, window_start + phase_last - start_frame)
                    if start <= phase_end:
                        frame_previous_targets[start:phase_end + 1] = all_token_ids[token_idx - 1]

        return {
            "video": mouth,
            "fps": fps,
            "teacher": torch.tensor(teacher, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
            "token_video_end": torch.tensor(token_video_end, dtype=torch.long),
            "text_available_at": torch.tensor(text_available_at, dtype=torch.long),
            "windows": torch.tensor(target_windows, dtype=torch.long).reshape(-1, 2),
            "query_start": prefix_count,
            "window_count": len(token_ids),
            "text": text,
            "path": row["video"],
            "sample_index": idx,
            "frame_targets": frame_targets,
            "frame_previous_targets": frame_previous_targets,
            "frame_windows": torch.tensor(frame_windows, dtype=torch.long).reshape(-1, 2),
        }


def make_collate(pad_id: int):
    def collate(batch: Sequence[dict]) -> dict:
        B = len(batch)
        max_f = max(x["video"].shape[0] for x in batch)
        max_t = max(x["teacher"].shape[0] for x in batch)
        # Keep a masked placeholder for batches with no token targets.
        max_q = max(1, max(x["targets"].shape[0] for x in batch))
        C, H, W = batch[0]["video"].shape[1:]
        video = batch[0]["video"].new_zeros(B, max_f, C, H, W)
        teacher = torch.full((B, max_t), pad_id, dtype=torch.long)
        targets = torch.full((B, max_q), pad_id, dtype=torch.long)
        token_video_end = torch.zeros(B, max_q, dtype=torch.long)
        text_available_at = torch.zeros(B, max_t, dtype=torch.long)
        windows = torch.zeros(B, max_q, 2, dtype=torch.long)
        video_lengths = torch.zeros(B, dtype=torch.long)
        text_lengths = torch.zeros(B, dtype=torch.long)
        query_lengths = torch.zeros(B, dtype=torch.long)
        query_starts = torch.zeros(B, dtype=torch.long)
        window_counts = torch.zeros(B, dtype=torch.long)
        frame_targets = torch.full((B, max_f), pad_id, dtype=torch.long)
        frame_previous_targets = torch.full_like(frame_targets, pad_id)
        max_fw = max(1, max(len(x.get("frame_windows", [])) for x in batch))
        frame_windows = torch.zeros(B, max_fw, 2, dtype=torch.long)
        frame_window_counts = torch.zeros(B, dtype=torch.long)
        for b, item in enumerate(batch):
            fv = item["video"].shape[0]
            tt = item["teacher"].shape[0]
            tq = item["targets"].shape[0]
            video[b, :fv] = item["video"]
            teacher[b, :tt] = item["teacher"]
            targets[b, :tq] = item["targets"]
            token_video_end[b, :tq] = item["token_video_end"]
            text_available_at[b, :tt] = item["text_available_at"]
            windows[b, :tq] = item["windows"]
            video_lengths[b] = fv
            text_lengths[b] = tt
            query_lengths[b] = tq
            query_starts[b] = item.get("query_start", 0)
            window_counts[b] = item.get("window_count", max(0, tq - 1))
            if "frame_targets" in item:
                frame_targets[b, :fv] = item["frame_targets"]
                if "frame_previous_targets" in item:
                    frame_previous_targets[b, :fv] = item["frame_previous_targets"]
                fw = len(item["frame_windows"])
                frame_windows[b, :fw] = item["frame_windows"]
                frame_window_counts[b] = fw
        return {
            "video": video,
            "teacher": teacher,
            "targets": targets,
            "token_video_end": token_video_end,
            "text_available_at": text_available_at,
            "windows": windows,
            "video_lengths": video_lengths,
            "text_lengths": text_lengths,
            "query_lengths": query_lengths,
            "query_starts": query_starts,
            "window_counts": window_counts,
            "paths": [x["path"] for x in batch],
            "texts": [x["text"] for x in batch],
            "sample_indices": [x.get("sample_index") for x in batch],
            "frame_targets": frame_targets,
            "frame_previous_targets": frame_previous_targets,
            "frame_windows": frame_windows,
            "frame_window_counts": frame_window_counts,
        }

    return collate


def batch_query_lengths(batch: dict) -> torch.Tensor:
    """Supervised queries, distinct from the complete teacher prefix length."""
    return batch.get("query_lengths", batch["text_lengths"])


def batch_window_counts(batch: dict) -> torch.Tensor:
    """Real token windows; legacy full-video batches have one trailing EOS."""
    return batch.get("window_counts", (batch_query_lengths(batch) - 1).clamp_min(0))


def select_text_queries(x_text: torch.Tensor, batch: dict) -> torch.Tensor:
    """Gather section queries from globally contextualized teacher states."""
    if "query_starts" not in batch:
        return x_text
    positions = batch["query_starts"][:, None] + torch.arange(
        batch["targets"].shape[1], device=x_text.device
    )[None, :]
    # Padding/empty queries need a legal gather index but remain loss-masked.
    positions = torch.minimum(positions, (batch["text_lengths"] - 1)[:, None])
    return x_text.gather(1, positions[..., None].expand(-1, -1, x_text.shape[-1]))


# -----------------------------------------------------------------------------
# Attention with both full-sequence and incremental KV-cache paths
# -----------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.d_model = d_model
        self.heads = heads
        self.d_head = d_model // heads
        self.scale = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.heads, self.d_head).transpose(1, 2)  # [B,H,T,Dh]

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def project_kv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._split(self.k_proj(x)), self._split(self.v_proj(x))

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allowed: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # q: [B,H,Tq,Dh]; k/v: [B,H,Tk,Dh]. allowed: [B,Tq,Tk] or
        # [Tq,Tk], with True marking visible keys.
        if not need_weights:
            # SDPA chooses the available backend; fused backends avoid dense
            # [B,H,Tq,Tk] scores/probabilities. Use the explicit path below when
            # callers request attention weights.
            attn_mask = allowed
            if attn_mask is not None:
                if attn_mask.ndim == 2:
                    attn_mask = attn_mask.unsqueeze(0)
                attn_mask = attn_mask.unsqueeze(1)
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
                scale=self.scale,
            )
            out = self.out_proj(self._merge(out))
            return out, None

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if allowed is not None:
            if allowed.ndim == 2:
                allowed = allowed.unsqueeze(0)
            scores = scores.masked_fill(~allowed.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        # Padded query rows can have no legal keys; keep them numerically benign.
        if allowed is not None:
            any_key = allowed.any(dim=-1, keepdim=True).unsqueeze(1)
            weights = torch.where(any_key, weights, torch.zeros_like(weights))
        # Losses consume normalized pre-dropout probabilities. Apply dropout
        # only to the value aggregation, so it cannot erase frame mass from
        # the monotonic or aligned-window attention objectives.
        out = torch.matmul(self.dropout(weights), v)
        out = self.out_proj(self._merge(out))
        return out, weights if need_weights else None

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        allowed: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self._split(self.q_proj(query))
        k, v = self.project_kv(key_value)
        return self._attend(q, k, v, allowed=allowed, need_weights=need_weights)

    def forward_with_qk(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        allowed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use SDPA for values and expose projected Q/K for streamed losses."""
        q = self._split(self.q_proj(query))
        k, v = self.project_kv(key_value)
        out, _ = self._attend(q, k, v, allowed=allowed, need_weights=False)
        return out, q, k

    def self_attention(
        self,
        x: torch.Tensor,
        valid: Optional[torch.Tensor] = None,
        causal: bool = True,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, _ = x.shape
        allowed = None
        if causal:
            allowed = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            allowed = allowed.unsqueeze(0).expand(B, -1, -1)
        if valid is not None:
            key_ok = valid[:, None, :]
            allowed = key_ok if allowed is None else (allowed & key_ok)
        return self.forward(x, x, allowed=allowed, need_weights=need_weights)

    def step_self(
        self, x_t: torch.Tensor, cache: Optional[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x_t: [B,1,D]. Append its K/V, then attend to all cached past/current.
        q = self._split(self.q_proj(x_t))
        k_new, v_new = self.project_kv(x_t)
        if cache is None or cache.get("k") is None:
            k, v = k_new, v_new
        else:
            k = torch.cat([cache["k"], k_new], dim=2)
            v = torch.cat([cache["v"], v_new], dim=2)
        out, _ = self._attend(q, k, v)
        return out, {"k": k, "v": v}

    def append_kv(
        self, x_t: torch.Tensor, cache: Optional[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        k_new, v_new = self.project_kv(x_t)
        if cache is None or cache.get("k") is None:
            return {"k": k_new, "v": v_new}
        return {
            "k": torch.cat([cache["k"], k_new], dim=2),
            "v": torch.cat([cache["v"], v_new], dim=2),
        }

    def query_cached(
        self,
        query: torch.Tensor,
        cache: Dict[str, torch.Tensor],
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if cache is None or cache.get("k") is None or cache["k"].shape[2] == 0:
            raise RuntimeError("Cross-attention KV cache is empty")
        q = self._split(self.q_proj(query))
        return self._attend(q, cache["k"], cache["v"], need_weights=need_weights)


# -----------------------------------------------------------------------------
# Causal visual encoder
# -----------------------------------------------------------------------------


def sinusoidal_position(length: int, dim: int, device, dtype, offset: int = 0) -> torch.Tensor:
    pos = torch.arange(offset, offset + length, device=device, dtype=torch.float32).unsqueeze(1)
    half = (dim + 1) // 2
    div = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1)))
    angles = pos * div.unsqueeze(0)
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(angles[:, : pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(angles[:, : pe[:, 1::2].shape[1]])
    return pe.to(dtype=dtype)


class CausalConv3d(nn.Module):
    """Temporal-left-only padding; centered spatial padding as in the draft."""

    def __init__(self, in_ch: int, out_ch: int, kernel=(5, 5, 5), stride=(1, 2, 2)):
        super().__init__()
        self.kernel = tuple(kernel)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=0, bias=False)
        self.norm = AutocastGroupNorm(8 if out_ch % 8 == 0 else 1, out_ch)

    def _pad(self, x: torch.Tensor, temporal: bool = True) -> torch.Tensor:
        kt, kh, kw = self.kernel
        pt = kt - 1 if temporal else 0
        ph, pw = kh - 1, kw - 1
        hl, hr = ph // 2, ph - ph // 2
        wl, wr = pw // 2, pw - pw // 2
        return F.pad(x, (wl, wr, hl, hr, pt, 0))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # [B,T,C,H,W] -> [B,T,C',H',W']
        x = video.transpose(1, 2)
        x = self._pad(x, temporal=True)
        x = self.conv(x)  # [B,C,T,H,W]
        # GroupNorm on a 5-D tensor would aggregate statistics across time and
        # therefore make training differ from one-frame streaming. Normalize
        # each temporal slice independently instead.
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        x = F.silu(self.norm(x))
        return x.view(B, T, C, H, W)

    def step(
        self, frame: torch.Tensor, cache: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # frame [B,C,H,W], cache stores previous raw input frames [B,C,Kt-1,H,W].
        kt = self.kernel[0]
        cur = frame.unsqueeze(2)
        if kt > 1:
            if cache is None:
                cache = torch.zeros(
                    frame.shape[0], frame.shape[1], kt - 1, frame.shape[2], frame.shape[3],
                    device=frame.device, dtype=frame.dtype,
                )
            x = torch.cat([cache, cur], dim=2)
            new_cache = x[:, :, -(kt - 1) :].detach()
        else:
            x = cur
            new_cache = cur[:, :, :0].detach()
        x = self._pad(x, temporal=False)
        z = self.conv(x).squeeze(2)  # [B,C',H',W']
        z = F.silu(self.norm(z))
        return z, new_cache


class Residual2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        g1 = 8 if out_ch % 8 == 0 else 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = AutocastGroupNorm(g1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = AutocastGroupNorm(g1, out_ch)
        self.skip = nn.Identity() if (in_ch == out_ch and stride == 1) else nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class FrameVisualEncoder(nn.Module):
    def __init__(self, in_ch: int, d_video: int):
        super().__init__()
        self.net = nn.Sequential(
            Residual2D(in_ch, 64, stride=2),
            Residual2D(64, 96, stride=2),
            Residual2D(96, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, d_video)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,T,C,H,W]
        B, T, C, H, W = x.shape
        h = self.net(x.reshape(B * T, C, H, W)).flatten(1)
        return self.proj(h).view(B, T, -1)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.proj(h).unsqueeze(1)


class FFN(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float):
        super().__init__()
        self.norm = AutocastLayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class CausalConv1dModule(nn.Module):
    def __init__(self, d_model: int, kernel: int, dropout: float):
        super().__init__()
        self.kernel = kernel
        self.norm = AutocastLayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dw = nn.Conv1d(d_model, d_model, kernel, groups=d_model, padding=0)
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h = F.glu(self.pw1(h), dim=1)
        h = F.pad(h, (self.kernel - 1, 0))
        h = F.silu(self.dw(h))
        h = self.pw2(h).transpose(1, 2)
        return self.dropout(h)

    def step(self, x_t: torch.Tensor, cache: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.norm(x_t).transpose(1, 2)
        h = F.glu(self.pw1(h), dim=1)  # [B,D,1]
        if self.kernel > 1:
            if cache is None:
                cache = torch.zeros(h.shape[0], h.shape[1], self.kernel - 1, device=h.device, dtype=h.dtype)
            cat = torch.cat([cache, h], dim=-1)
            new_cache = cat[:, :, -(self.kernel - 1) :].detach()
        else:
            cat = h
            new_cache = h[:, :, :0].detach()
        y = F.silu(self.dw(cat))
        y = self.dropout(self.pw2(y).transpose(1, 2))
        return y, new_cache


class CausalConformerLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_mult: int, conv_kernel: int, dropout: float):
        super().__init__()
        self.ffn = FFN(d_model, ff_mult, dropout)
        self.attn_norm = AutocastLayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = CausalConv1dModule(d_model, conv_kernel, dropout)

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + 0.5 * self.ffn(x)
        a, _ = self.attn.self_attention(self.attn_norm(x), valid=valid, causal=True)
        x = x + self.attn_drop(a)
        x = x + self.conv(x)
        return x

    def step(self, x_t: torch.Tensor, state: Optional[dict]) -> Tuple[torch.Tensor, dict]:
        state = {} if state is None else state
        x_t = x_t + 0.5 * self.ffn(x_t)
        a, attn_cache = self.attn.step_self(self.attn_norm(x_t), state.get("attn"))
        x_t = x_t + self.attn_drop(a)
        c, conv_cache = self.conv.step(x_t, state.get("conv"))
        x_t = x_t + c
        return x_t, {"attn": attn_cache, "conv": conv_cache}


class VideoEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.front = CausalConv3d(3, cfg.conv3d_channels, kernel=(5, 5, 5), stride=(1, 2, 2))
        self.frame_encoder = FrameVisualEncoder(cfg.conv3d_channels, cfg.d_video)
        self.layers = nn.ModuleList([
            CausalConformerLayer(cfg.d_video, cfg.heads, cfg.ff_mult, cfg.conv_kernel, cfg.dropout)
            for _ in range(cfg.video_layers)
        ])
        self.out_norm = AutocastLayerNorm(cfg.d_video)
        self.d_video = cfg.d_video

    def _encode_frames(self, video: torch.Tensor) -> torch.Tensor:
        return self.frame_encoder(self.front(video))

    def forward(self, video: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled() and torch.is_autocast_enabled(video.device.type):
            # Recompute the large spatial maps in backward. Splitting frames
            # alone would still retain every chunk's activation graph.
            x = checkpoint(self._encode_frames, video, use_reentrant=False)
        else:
            x = self._encode_frames(video)
        x = x + sinusoidal_position(x.shape[1], self.d_video, x.device, x.dtype).unsqueeze(0)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        for layer in self.layers:
            x = layer(x, valid)
        return self.out_norm(x)

    def init_stream_state(self) -> dict:
        return {"front": None, "layers": [None] * len(self.layers), "pos": 0}

    def step(self, frame: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        z, front_cache = self.front.step(frame, state.get("front"))
        x = self.frame_encoder.step(z)
        pos = int(state.get("pos", 0))
        x = x + sinusoidal_position(1, self.d_video, x.device, x.dtype, offset=pos).unsqueeze(0)
        new_layers = []
        for i, layer in enumerate(self.layers):
            x, layer_state = layer.step(x, state["layers"][i])
            new_layers.append(layer_state)
        x = self.out_norm(x)
        return x, {"front": front_cache, "layers": new_layers, "pos": pos + 1}


# -----------------------------------------------------------------------------
# Causal text encoder
# -----------------------------------------------------------------------------


class CausalTextLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.attn_norm = AutocastLayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FFN(d_model, ff_mult, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn.self_attention(self.attn_norm(x), valid=valid, causal=True)
        x = x + self.dropout(a)
        x = x + self.ffn(x)
        return x

    def step(self, x_t: torch.Tensor, cache: Optional[dict]) -> Tuple[torch.Tensor, dict]:
        a, new_cache = self.attn.step_self(self.attn_norm(x_t), cache)
        x_t = x_t + self.dropout(a)
        x_t = x_t + self.ffn(x_t)
        return x_t, new_cache


class TextEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_text)
        self.layers = nn.ModuleList([
            CausalTextLayer(cfg.d_text, cfg.heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.text_layers)
        ])
        self.out_norm = AutocastLayerNorm(cfg.d_text)
        self.d_text = cfg.d_text

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = autocast_activation(self.embed(ids))
        x = x + sinusoidal_position(x.shape[1], self.d_text, x.device, x.dtype).unsqueeze(0)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        for layer in self.layers:
            x = layer(x, valid)
        return self.out_norm(x)

    def init_stream_state(self) -> dict:
        return {"layers": [None] * len(self.layers), "pos": 0}

    def step(self, token_id: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        # token_id [B] or [B,1]
        if token_id.ndim == 1:
            token_id = token_id[:, None]
        pos = int(state.get("pos", 0))
        x = autocast_activation(self.embed(token_id))
        x = x + sinusoidal_position(1, self.d_text, x.device, x.dtype, offset=pos).unsqueeze(0)
        new_layers = []
        for i, layer in enumerate(self.layers):
            x, cache = layer.step(x, state["layers"][i])
            new_layers.append(cache)
        x = self.out_norm(x)
        return x, {"layers": new_layers, "pos": pos + 1}


class PretrainedLMTextEncoder(nn.Module):
    """Decoder-only Hugging Face LM used as CESM's causal text encoder.

    The LM head is intentionally discarded: CESM's Mixed Block remains the
    next-token predictor, while the pretrained transformer supplies contextual
    text states. Full-sequence training disables caching; one-token streaming
    uses the same model with its native ``past_key_values`` cache.
    """

    def __init__(self, cfg: ModelConfig, *, initialize_from_pretrained: bool):
        super().__init__()
        _, AutoConfig, AutoModelForCausalLM, _, _ = _require_huggingface()
        if initialize_from_pretrained:
            if not cfg.pretrained_lm_name_or_path:
                raise ValueError("pretrained_lm_name_or_path is required for a pretrained text encoder")
            causal_lm = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_lm_name_or_path,
                local_files_only=cfg.pretrained_lm_local_files_only,
            )
            causal_lm.resize_token_embeddings(cfg.vocab_size)
            # Save the post-resize architecture so inference/resume can rebuild
            # it without reaching the model hub.
            cfg.pretrained_lm_config = causal_lm.config.to_dict()
        else:
            if not cfg.pretrained_lm_config:
                raise ValueError(
                    "Checkpoint is missing pretrained_lm_config; cannot rebuild its LM offline"
                )
            config_data = dict(cfg.pretrained_lm_config)
            model_type = config_data.pop("model_type", None)
            if not model_type:
                raise ValueError("Checkpoint Hugging Face config is missing model_type")
            try:
                hf_config = AutoConfig.for_model(model_type, **config_data)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported Hugging Face model_type in checkpoint: {model_type!r}"
                ) from exc
            causal_lm = AutoModelForCausalLM.from_config(hf_config)

        self.model = causal_lm.base_model
        hidden_size = int(getattr(causal_lm.config, "hidden_size"))
        if hidden_size != cfg.d_text:
            raise ValueError(
                f"Pretrained LM hidden size is {hidden_size}, but ModelConfig.d_text={cfg.d_text}"
            )
        self.d_text = hidden_size
        self.frozen = bool(cfg.freeze_text_encoder)
        if self.frozen:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen imported LM is a deterministic feature extractor; keep its
        # dropout disabled while the CESM video/fusion modules train.
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        attention_mask = positions[None, :] < lengths[:, None]
        if self.frozen:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        else:
            outputs = self.model(
                input_ids=ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        return outputs.last_hidden_state

    def init_stream_state(self) -> dict:
        return {"past_key_values": None, "attention_mask": None, "pos": 0}

    def step(self, token_id: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        if token_id.ndim == 1:
            token_id = token_id[:, None]
        if token_id.ndim != 2 or token_id.shape[1] != 1:
            raise ValueError("PretrainedLMTextEncoder.step expects one token per batch item")
        prior_mask = state.get("attention_mask")
        current_mask = torch.ones(
            token_id.shape[0], 1, dtype=torch.bool, device=token_id.device
        )
        attention_mask = (
            current_mask if prior_mask is None else torch.cat([prior_mask, current_mask], dim=1)
        )
        outputs = self.model(
            input_ids=token_id,
            attention_mask=attention_mask,
            past_key_values=state.get("past_key_values"),
            use_cache=True,
            return_dict=True,
        )
        return outputs.last_hidden_state, {
            "past_key_values": outputs.past_key_values,
            "attention_mask": attention_mask,
            "pos": int(state.get("pos", 0)) + 1,
        }


# -----------------------------------------------------------------------------
# Mixed block + streaming cross-attention caches
# -----------------------------------------------------------------------------


class MixedBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Without vproj, every fusion operation lives in the encoder's video
        # feature space; text must be mapped to that same width.
        df = cfg.d_fusion if cfg.use_video_projection else cfg.d_video
        self.vproj = nn.Linear(cfg.d_video, df, bias=False) if cfg.use_video_projection else nn.Identity()
        self.tproj = nn.Linear(cfg.d_text, df, bias=False)
        self.text_cross = (
            None if cfg.text_fusion == "frame" else MultiHeadAttention(df, cfg.heads, cfg.dropout)
        )
        self.video_cross = MultiHeadAttention(df, cfg.heads, cfg.dropout)
        self.text_ln = AutocastLayerNorm(df)
        self.video_ln = AutocastLayerNorm(df)
        self.text_ffn = FFN(df, cfg.ff_mult, cfg.dropout)
        self.text_ffn_ln = AutocastLayerNorm(df)
        self.video_self_norm = AutocastLayerNorm(df)
        self.video_self = MultiHeadAttention(df, cfg.heads, cfg.dropout)
        self.video_self_ln = AutocastLayerNorm(df)
        self.video_ffn = FFN(df, cfg.ff_mult, cfg.dropout)
        self.video_ffn_ln = AutocastLayerNorm(df)
        self.token_head = nn.Linear(df, cfg.vocab_size)
        self.conf_head = nn.Linear(df, 1)
        if cfg.text_fusion not in ("residual", "cross_only", "weighted", "frame"):
            raise ValueError(f"Unsupported text_fusion: {cfg.text_fusion!r}")
        self.text_fusion = cfg.text_fusion
        if not math.isfinite(cfg.frame_vcross_weight):
            raise ValueError("frame_vcross_weight must be finite")
        # Stored in ModelConfig, so older checkpoints need no new state tensor.
        self.frame_vcross_weight = cfg.frame_vcross_weight
        if self.text_fusion == "weighted":
            if not math.isfinite(cfg.text_residual_weight):
                raise ValueError("text_residual_weight must be finite")
            self.register_buffer("w_t", torch.tensor(float(cfg.text_residual_weight)))

    def fuse_text(self, ht0: torch.Tensor, tcross: torch.Tensor) -> torch.Tensor:
        """Use the same experiment in training, confidence targets and decoding."""
        if self.text_fusion == "cross_only":
            return self.text_ln(tcross)
        if self.text_fusion == "weighted":
            return self.text_ln(self.w_t * ht0 + tcross)
        return self.text_ln(ht0 + tcross)

    def frame_logits(self, hv0: torch.Tensor, vcross: torch.Tensor) -> torch.Tensor:
        """Classify frames; projected text contributes through video->text attention."""
        ht = self.text_ln(hv0 + self.frame_vcross_weight * vcross)
        ht_hat = self.text_ffn_ln(ht + self.text_ffn(ht))
        return self.token_head(ht_hat)

    def forward_frame(self, x_video, x_text, video_to_text_allowed) -> dict:
        hv0 = self.vproj(x_video)
        ht0 = self.project_text(x_text)
        attention = self.video_cross
        q = attention._split(attention.q_proj(hv0))
        k, v = attention.project_kv(ht0)
        vcross, _ = attention._attend(q, k, v, allowed=video_to_text_allowed)
        return {
            "token_logits": self.frame_logits(hv0, vcross),
            "video_projected": hv0,
            "text_projected": ht0,
            # Reuse these tensors for phased targets without another encoder
            # or Q/K/V projection pass. No extra tensor storage is allocated.
            "frame_q": q,
            "frame_k": k,
            "frame_v": v,
        }

    def project_text(self, x_text: torch.Tensor) -> torch.Tensor:
        """Cross the pretrained-LM/CESM precision boundary explicitly.

        Pretrained language models commonly emit float16 or bfloat16 states
        even when the newly initialized CESM projection remains float32.  A
        plain Linear call requires matching dtypes outside autocast.  Casting
        only the activations preserves the LM's memory-efficient parameter
        dtype and remains differentiable when LM fine-tuning is enabled.
        """
        if torch.is_autocast_enabled(x_text.device.type):
            return self.tproj(x_text)
        return self.tproj(x_text.to(dtype=self.tproj.weight.dtype))

    def forward(
        self,
        x_video: torch.Tensor,
        x_text: torch.Tensor,
        video_valid: torch.Tensor,
        text_valid: torch.Tensor,
        text_to_video_allowed: torch.Tensor,
        video_to_text_allowed: torch.Tensor,
    ) -> dict:
        if self.text_fusion == "frame":
            out = self.forward_frame(x_video, x_text, video_to_text_allowed)
            out["confidence_logits"] = self.forward_confidence(
                x_video, x_text, video_valid, video_to_text_allowed,
            )
            return out
        hv0 = self.vproj(x_video)
        ht0 = self.project_text(x_text)

        tcross, t_attn = self.text_cross(
            ht0, hv0, allowed=text_to_video_allowed, need_weights=True
        )
        ht = self.fuse_text(ht0, tcross)
        ht_hat = self.text_ffn_ln(ht + self.text_ffn(ht))

        vcross, _ = self.video_cross(hv0, ht0, allowed=video_to_text_allowed)
        hv = self.video_ln(hv0 + vcross)
        vself, _ = self.video_self.self_attention(
            self.video_self_norm(hv), valid=video_valid, causal=True
        )
        hv_prime = self.video_self_ln(hv + vself)
        hv_hat = self.video_ffn_ln(hv_prime + self.video_ffn(hv_prime))

        return {
            "token_logits": self.token_head(ht_hat),
            "confidence_logits": self.conf_head(hv_hat).squeeze(-1),
            "text_video_attn": t_attn,
            "video_projected": hv0,
            "text_projected": ht0,
            "tcross": tcross,
        }

    def forward_token(
        self,
        x_video: torch.Tensor,
        x_text: torch.Tensor,
        text_to_video_allowed: torch.Tensor,
        *,
        flash_mono: bool = False,
        need_attention: bool = True,
    ) -> dict:
        """Run the token-query branch for non-frame variants.

        These variants' video/confidence branch does not contribute to the
        first-stage losses, so skip its unused graph. Frame fusion uses
        forward_frame instead because video->text attention also predicts tokens.
        """
        hv0 = self.vproj(x_video)
        ht0 = self.project_text(x_text)
        if not need_attention and torch.is_autocast_enabled(x_video.device.type):
            tcross, _ = self.text_cross(ht0, hv0, allowed=text_to_video_allowed)
            attention = {}
        elif flash_mono:
            tcross, q, k = self.text_cross.forward_with_qk(
                ht0, hv0, allowed=text_to_video_allowed
            )
            attention = {"text_video_q": q, "text_video_k": k}
        else:
            tcross, t_attn = self.text_cross(
                ht0, hv0, allowed=text_to_video_allowed, need_weights=True
            )
            attention = {"text_video_attn": t_attn}
        ht = self.fuse_text(ht0, tcross)
        ht_hat = self.text_ffn_ln(ht + self.text_ffn(ht))
        return {
            "token_logits": self.token_head(ht_hat),
            **attention,
            "video_projected": hv0,
            "text_projected": ht0,
            "tcross": tcross,
        }

    def forward_confidence(
        self,
        x_video: torch.Tensor,
        x_text: torch.Tensor,
        video_valid: torch.Tensor,
        video_to_text_allowed: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the document's video/confidence branch.

        This is used for the second training stage after freezing the encoders,
        projections and token-prediction branch. It avoids spending compute on
        text->video token logits that cannot receive gradients in that stage.
        """
        hv0 = self.vproj(x_video)
        ht0 = self.project_text(x_text)
        vcross, _ = self.video_cross(hv0, ht0, allowed=video_to_text_allowed)
        hv = self.video_ln(hv0 + vcross)
        vself, _ = self.video_self.self_attention(
            self.video_self_norm(hv), valid=video_valid, causal=True
        )
        hv_prime = self.video_self_ln(hv + vself)
        hv_hat = self.video_ffn_ln(hv_prime + self.video_ffn(hv_prime))
        return self.conf_head(hv_hat).squeeze(-1)

    def init_stream_state(self) -> dict:
        return {
            "text_to_video_kv": None,  # K/V from projected video, queried by text
            "video_to_text_kv": None,  # K/V from projected text, queried by video
            "video_self": None,
        }

    def append_text(self, x_text_t: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        ht0 = self.project_text(x_text_t)
        new = dict(state)
        new["video_to_text_kv"] = self.video_cross.append_kv(ht0, state.get("video_to_text_kv"))
        return ht0, new

    def append_video(self, x_video_t: torch.Tensor, state: dict) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        hv0 = self.vproj(x_video_t)
        new = dict(state)
        if self.text_fusion != "frame":
            new["text_to_video_kv"] = self.text_cross.append_kv(hv0, state.get("text_to_video_kv"))

        # Confidence/video path is causal over the text states that have actually
        # been created so far by the decoder.
        if state.get("video_to_text_kv") is None:
            vcross = torch.zeros_like(hv0)
        else:
            vcross, _ = self.video_cross.query_cached(hv0, state["video_to_text_kv"])
        if self.text_fusion == "frame":
            new["frame_token_logits"] = self.frame_logits(hv0, vcross)
        hv = self.video_ln(hv0 + vcross)
        vself, video_self_cache = self.video_self.step_self(
            self.video_self_norm(hv), state.get("video_self")
        )
        hv_prime = self.video_self_ln(hv + vself)
        hv_hat = self.video_ffn_ln(hv_prime + self.video_ffn(hv_prime))
        conf = self.conf_head(hv_hat).squeeze(-1)
        new["video_self"] = video_self_cache
        return hv0, conf, new

    def predict_text_query(
        self, projected_text_query: torch.Tensor, state: dict, need_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.text_fusion == "frame":
            if "frame_token_logits" not in state:
                raise RuntimeError("Push a video frame before requesting its token prediction")
            return state["frame_token_logits"], None
        tcross, weights = self.text_cross.query_cached(
            projected_text_query, state["text_to_video_kv"], need_weights=need_weights
        )
        ht = self.fuse_text(projected_text_query, tcross)
        ht_hat = self.text_ffn_ln(ht + self.text_ffn(ht))
        return self.token_head(ht_hat), weights


# -----------------------------------------------------------------------------
# Complete model and masks
# -----------------------------------------------------------------------------


class UnnobaModel(nn.Module):
    def __init__(self, cfg: ModelConfig, *, initialize_pretrained_text_encoder: bool = False):
        super().__init__()
        self.cfg = cfg
        self.video_encoder = VideoEncoder(cfg)
        if cfg.text_encoder_type == "learned":
            self.text_encoder = TextEncoder(cfg)
        elif cfg.text_encoder_type == "pretrained_lm":
            self.text_encoder = PretrainedLMTextEncoder(
                cfg, initialize_from_pretrained=initialize_pretrained_text_encoder
            )
        else:
            raise ValueError(f"Unsupported text_encoder_type: {cfg.text_encoder_type!r}")
        self.mixed = MixedBlock(cfg)

    @staticmethod
    def build_masks(
        video_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        token_video_end: torch.Tensor,
        text_available_at: torch.Tensor,
        max_f: int,
        max_t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = video_lengths.device
        frames = torch.arange(max_f, device=device)
        toks = torch.arange(max_t, device=device)
        video_valid = frames[None, :] < video_lengths[:, None]
        text_valid = toks[None, :] < text_lengths[:, None]

        # Text query t predicts token t and may see only video available by the
        # end of that token's aligned frame window.
        t2v = frames[None, None, :] <= token_video_end[:, :, None]
        t2v = t2v & video_valid[:, None, :] & text_valid[:, :, None]

        # Video frame f can only see teacher text states that would have been
        # created by f in streaming mode.
        v2t = text_available_at[:, None, :] <= frames[None, :, None]
        v2t = v2t & text_valid[:, None, :] & video_valid[:, :, None]
        return video_valid, text_valid, t2v, v2t

    @staticmethod
    def build_token_masks(
        video_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        token_video_end: torch.Tensor,
        max_f: int,
        max_t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build validity and text->video masks for token-query variants."""
        device = video_lengths.device
        frames = torch.arange(max_f, device=device)
        toks = torch.arange(max_t, device=device)
        video_valid = frames[None, :] < video_lengths[:, None]
        text_valid = toks[None, :] < text_lengths[:, None]
        t2v = frames[None, None, :] <= token_video_end[:, :, None]
        t2v = t2v & video_valid[:, None, :] & text_valid[:, :, None]
        return video_valid, text_valid, t2v

    @staticmethod
    def build_confidence_masks(
        video_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        text_available_at: torch.Tensor,
        max_f: int,
        max_t: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Confidence sees all previously committed text, including history
        # preceding this video section. It is not limited to supervised queries.
        frames = torch.arange(max_f, device=video_lengths.device)
        toks = torch.arange(max_t, device=video_lengths.device)
        video_valid = frames[None, :] < video_lengths[:, None]
        text_valid = toks[None, :] < text_lengths[:, None]
        v2t = text_available_at[:, None, :] <= frames[None, :, None]
        v2t = v2t & text_valid[:, None, :] & video_valid[:, :, None]
        return video_valid, v2t

    def forward(self, batch: dict, *, flash_mono: bool = False, need_attention: bool = True) -> dict:
        x_video = self.video_encoder(batch["video"], batch["video_lengths"])
        x_text = self.text_encoder(batch["teacher"], batch["text_lengths"])
        if self.cfg.text_fusion == "frame":
            vv, v2t = self.build_confidence_masks(
                batch["video_lengths"], batch["text_lengths"], batch["text_available_at"],
                x_video.shape[1], x_text.shape[1],
            )
            tv = torch.arange(x_text.shape[1], device=x_text.device)[None] < batch["text_lengths"][:, None]
            out = self.mixed.forward_frame(x_video, x_text, v2t)
        else:
            x_text = select_text_queries(x_text, batch)
            vv, tv, t2v = self.build_token_masks(
                batch["video_lengths"], batch_query_lengths(batch), batch["token_video_end"],
                x_video.shape[1], x_text.shape[1]
            )
            out = self.mixed.forward_token(
                x_video, x_text, t2v, flash_mono=flash_mono, need_attention=need_attention,
            )
        out["video_encoded"] = x_video
        out["text_encoded"] = x_text
        out["video_valid"] = vv
        out["text_valid"] = tv
        return out

    def init_stream_state(self) -> dict:
        return {
            "video": self.video_encoder.init_stream_state(),
            "text": self.text_encoder.init_stream_state(),
            "mixed": self.mixed.init_stream_state(),
            "current_text_query": None,
        }

    def stream_push_text_token(self, token_id: torch.Tensor, state: dict) -> dict:
        xt, text_state = self.text_encoder.step(token_id, state["text"])
        projected, mixed_state = self.mixed.append_text(xt, state["mixed"])
        new = dict(state)
        new["text"] = text_state
        new["mixed"] = mixed_state
        new["current_text_query"] = projected
        return new

    def stream_push_video_frame(self, frame: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        xv, video_state = self.video_encoder.step(frame, state["video"])
        _, conf, mixed_state = self.mixed.append_video(xv, state["mixed"])
        new = dict(state)
        new["video"] = video_state
        new["mixed"] = mixed_state
        return conf, new

    def stream_predict_next(self, state: dict) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if state["current_text_query"] is None:
            raise RuntimeError("Push BOS/text token before requesting a prediction")
        return self.mixed.predict_text_query(state["current_text_query"], state["mixed"], need_weights=True)

    def forward_confidence(self, batch: dict) -> torch.Tensor:
        """Confidence-only forward used after freezing the token model."""
        x_video = self.video_encoder(batch["video"], batch["video_lengths"])
        x_text = self.text_encoder(batch["teacher"], batch["text_lengths"])
        vv, v2t = self.build_confidence_masks(
            batch["video_lengths"], batch["text_lengths"],
            batch["text_available_at"], x_video.shape[1], x_text.shape[1]
        )
        return self.mixed.forward_confidence(x_video, x_text, vv, v2t)


# -----------------------------------------------------------------------------
# Losses from the draft + document confidence objective
# -----------------------------------------------------------------------------


def temporal_visual_loss(z: torch.Tensor, lengths: torch.Tensor, adjacent_frames: int = 2) -> torch.Tensor:
    """Mean feature MSE over unique valid pairs with 0 < j-i < adjacent_frames."""
    if adjacent_frames < 2:
        raise ValueError("--pretrain-adjacent-frames must be >= 2")
    z = z.float()
    valid = torch.arange(z.shape[1], device=z.device)[None, :] < lengths[:, None]
    # An empty slice gives a differentiable zero even with no valid pairs.
    total = z[:, :0].sum()
    count = lengths.new_zeros(())
    for offset in range(1, min(adjacent_frames, z.shape[1])):
        pairs = valid[:, :-offset] & valid[:, offset:]
        # Select before arithmetic so padded representations cannot contribute.
        errors = (z[:, :-offset][pairs] - z[:, offset:][pairs]).square().mean(-1)
        total = total + errors.sum()
        count = count + pairs.sum()
    return total / count.clamp_min(1)


def augmentation_visual_loss(z1: torch.Tensor, z2: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean feature MSE over valid frames, excluding batch padding."""
    valid = torch.arange(z1.shape[1], device=z1.device)[None, :] < lengths[:, None]
    errors = (z1[valid].float() - z2[valid].float()).square().mean(-1)
    return errors.sum() / valid.sum().clamp_min(1)


def visual_variance_covariance_losses(
    z: torch.Tensor, lengths: torch.Tensor, variance_floor: float = 1.0, *,
    compute_variance: bool = True, compute_covariance: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VICReg statistics over valid frames pooled across clips, in FP32.

    Use sample variance/covariance (N-1 denominator). Fewer than two valid
    frames cannot estimate diversity, so both terms are differentiable zeros.
    The floor is a per-feature standard-deviation target, not a variance target.
    """
    valid = torch.arange(z.shape[1], device=z.device)[None, :] < lengths[:, None]
    with torch.autocast(device_type=z.device.type, enabled=False):
        features = z[valid].float()
        zero = features[:0].sum()
        variance, covariance = zero, zero
        if features.shape[0] < 2:
            return variance, covariance
        if compute_variance:
            # Scale epsilon down for very small floors so a constant embedding
            # still incurs a penalty. The usual floor=1 uses VICReg's 1e-4.
            eps = min(1e-4, (0.01 * variance_floor) ** 2)
            std = torch.sqrt(features.var(dim=0) + eps)
            variance = F.relu(variance_floor - std).mean()
        if compute_covariance:
            centered = features - features.mean(dim=0)
            cov = (centered.T @ centered) / (features.shape[0] - 1)
            diagonal = torch.eye(cov.shape[0], dtype=torch.bool, device=cov.device)
            covariance = cov.masked_fill(diagonal, 0).square().sum() / cov.shape[0]
    return variance, covariance


@torch.no_grad()
def augment_pretrain_video(video: torch.Tensor, lengths: torch.Tensor,
                           original_probability: float = 0.2) -> torch.Tensor:
    """Independently jitter each valid frame; sometimes retain it exactly.

    Inputs/outputs are RGB in [-1,1]. Each call resamples brightness/contrast
    factors in [0.8,1.2], a rotation in [-5,5] degrees, and identity choices.
    """
    if not 0 <= original_probability <= 1:
        raise ValueError("original_probability must be in [0,1]")
    valid = torch.arange(video.shape[1], device=video.device)[None, :] < lengths[:, None]
    output = torch.zeros_like(video)
    frames = video[valid]
    if len(frames) == 0:
        return output
    # Grid sampling and color arithmetic use FP32 even when decoded frames
    # are stored in FP16/BF16 under AMP.
    with torch.autocast(device_type=video.device.type, enabled=False):
        pixels = (frames.float() + 1) / 2
        n = len(pixels)
        brightness = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, device=video.device)
        contrast = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, device=video.device)
        mean = pixels.mean(dim=(1, 2, 3), keepdim=True)
        pixels = ((pixels - mean) * contrast + mean) * brightness
        angle = (torch.rand(n, device=video.device) * 2 - 1) * math.radians(5)
        theta = pixels.new_zeros(n, 2, 3)
        theta[:, 0, 0] = theta[:, 1, 1] = angle.cos()
        theta[:, 0, 1] = -angle.sin()
        theta[:, 1, 0] = angle.sin()
        grid = F.affine_grid(theta, pixels.shape, align_corners=False)
        pixels = F.grid_sample(pixels, grid, padding_mode="border", align_corners=False)
        augmented = (pixels.clamp(0, 1) * 2 - 1).to(dtype=video.dtype)
        keep = torch.rand(n, 1, 1, 1, device=video.device) < original_probability
        output[valid] = torch.where(keep, frames, augmented)
    return output


def visual_pretraining_losses(encoder: VideoEncoder, batch: dict,
                             adjacent_frames: int = 2, *,
                             lambda_temporal: float = 0.0,
                             lambda_augmentation: float = 1.0,
                             lambda_variance: float = 1.0,
                             lambda_covariance: float = 0.04,
                             variance_floor: float = 1.0,
                             original_probability: float = 0.2) -> Dict[str, torch.Tensor]:
    """Weighted losses on final encoder states; disabled terms report zero."""
    video, lengths = batch["video"], batch["video_lengths"]
    # Avoid the third encoder pass when temporal smoothing is disabled.
    temporal = (temporal_visual_loss(encoder(video, lengths), lengths, adjacent_frames)
                if lambda_temporal else None)
    z1 = encoder(augment_pretrain_video(video, lengths, original_probability), lengths)
    z2 = encoder(augment_pretrain_video(video, lengths, original_probability), lengths)
    zero = z1[:, :0].float().sum() + z2[:, :0].float().sum()
    if temporal is None:
        temporal = zero
    augmentation = augmentation_visual_loss(z1, z2, lengths) if lambda_augmentation else zero
    variance, covariance = zero, zero
    if lambda_variance or lambda_covariance:
        v1, c1 = visual_variance_covariance_losses(
            z1, lengths, variance_floor,
            compute_variance=bool(lambda_variance), compute_covariance=bool(lambda_covariance),
        )
        v2, c2 = visual_variance_covariance_losses(
            z2, lengths, variance_floor,
            compute_variance=bool(lambda_variance), compute_covariance=bool(lambda_covariance),
        )
        variance = (v1 + v2) / 2
        covariance = c1 + c2
    total = (lambda_temporal * temporal + lambda_augmentation * augmentation
             + lambda_variance * variance + lambda_covariance * covariance)
    return {"total": total, "temporal": temporal, "augmentation": augmentation,
            "variance": variance, "covariance": covariance}


def similarity_preservation_loss(x: torch.Tensor, y: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    losses = []
    for b, L in enumerate(lengths.tolist()):
        if L <= 0:
            continue
        # Accumulate this cancellation-prone Gram identity in FP32 under AMP,
        # and scale by L before the final squared-norm reductions.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            xb = F.normalize(x[b, :L].float(), dim=-1, eps=1e-6)
            yb = F.normalize(y[b, :L].float(), dim=-1, eps=1e-6)
            # ||XX' - YY'||_F^2 / L^2
            #   = ||X'X/L||_F^2 + ||Y'Y/L||_F^2 - 2||X'Y/L||_F^2.
            # Pairwise matrices scale with feature widths, O(D^2), rather than
            # sequence length, O(L^2); normalized inputs still require O(L*D).
            inv_length = 1.0 / float(L)
            xx = (xb.transpose(0, 1) @ xb) * inv_length
            yy = (yb.transpose(0, 1) @ yb) * inv_length
            xy = (xb.transpose(0, 1) @ yb) * inv_length
            losses.append(
                xx.square().sum()
                + yy.square().sum()
                - 2.0 * xy.square().sum()
            )
    return torch.stack(losses).mean() if losses else x.new_zeros(())


def norm_preservation_loss(x: torch.Tensor, y: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled(x.device.type):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return norm_preservation_loss(x.float(), y.float(), lengths)
    losses = []
    for b, L in enumerate(lengths.tolist()):
        if L <= 0:
            continue
        nx = x[b, :L].norm(dim=-1)
        ny = y[b, :L].norm(dim=-1)
        losses.append(((ny / (nx + 1e-6) - 1.0) ** 2).mean())
    return torch.stack(losses).mean() if losses else x.new_zeros(())


def tcross_margin_loss(
    tcross: torch.Tensor, lengths: torch.Tensor, margin: float,
) -> torch.Tensor:
    """Mean ReLU(m - ||tcross||_2)^2, excluding padding and including true EOS.

    L2 over features, averaged over all valid token queries.
    Compute in FP32 regardless of AMP, retaining FP64 inputs for gradient checks.
    """
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("tcross margin must be finite and >= 0")
    with torch.amp.autocast(device_type=tcross.device.type, enabled=False):
        values = tcross.to(torch.float64 if tcross.dtype == torch.float64 else torch.float32)
        valid = torch.arange(values.shape[1], device=values.device)[None, :] < lengths[:, None]
        selected = values[valid]
        if selected.numel() == 0:
            return selected.sum() * 0.0
        norms = torch.linalg.vector_norm(selected, ord=2, dim=-1)
        return F.relu(margin - norms).square().mean()


def monotonic_attention_loss(
    attn: torch.Tensor, text_lengths: torch.Tensor, video_lengths: torch.Tensor,
    *, window_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # attn [B,H,T,F]. Average heads; exclude EOS from monotonic pairs.
    A = attn.mean(dim=1)
    losses = []
    counts = (text_lengths - 1).clamp_min(0) if window_counts is None else window_counts
    for b, (n, Fv) in enumerate(zip(counts.tolist(), video_lengths.tolist())):
        if n < 2:
            continue
        Ab = A[b, :n, :Fv]
        frame_idx = torch.arange(Fv, device=Ab.device, dtype=Ab.dtype)
        mu = (Ab * frame_idx[None, :]).sum(dim=-1)
        losses.append(F.relu(mu[:-1] - mu[1:]).mean())
    return torch.stack(losses).mean() if losses else A.new_zeros(())


def aligned_window_attention_loss(
    attn: torch.Tensor,
    windows: torch.Tensor,
    text_lengths: torch.Tensor,
    video_lengths: torch.Tensor,
    *, window_counts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean negative log attention mass inside real token windows; exclude EOS."""
    A = attn.mean(dim=1)
    vals = []
    counts = (text_lengths - 1).clamp_min(0) if window_counts is None else window_counts
    for b, (n, Fv) in enumerate(zip(counts.tolist(), video_lengths.tolist())):
        for t in range(n):
            s, e = windows[b, t].tolist()
            s = max(0, min(Fv - 1, s))
            e = max(s, min(Fv - 1, e))
            mass = A[b, t, s : e + 1].sum().clamp_min(1e-8)
            vals.append(-torch.log(mass))
    return torch.stack(vals).mean() if vals else A.new_zeros(())


def flash_attention_losses(
    q: torch.Tensor,
    k: torch.Tensor,
    token_video_end: torch.Tensor,
    text_lengths: torch.Tensor,
    video_lengths: torch.Tensor,
    windows: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = None,
    window_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-dropout monotonic/alignment losses without dense attention weights."""
    mu, mass = attention_statistics_from_qk(
        q, k, token_video_end, text_lengths, video_lengths, windows, scale=scale
    )
    monotonic, aligned = [], []
    counts = (text_lengths - 1).clamp_min(0) if window_counts is None else window_counts
    for b, n in enumerate(counts.tolist()):
        if n >= 2:
            monotonic.append(F.relu(mu[b, :n - 1] - mu[b, 1:n]).mean())
        if mass is not None and n:
            aligned.append(-mass[b, :n].clamp_min(1e-8).log())
    # Keep the no-pairs case differentiable (e.g. a batch of EOS-only sections).
    mono = torch.stack(monotonic).mean() if monotonic else mu.sum() * 0.0
    # Alignment averages across tokens; monotonic averages per eligible sample.
    align = torch.cat(aligned).mean() if aligned else mu.new_zeros(())
    return mono, align


def build_window_layout(
    windows: torch.Tensor,
    window_counts: torch.Tensor,
    max_frames: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the document's padded ``(window, timestep)`` indexing tensors."""
    B = windows.shape[0]
    Wmax = int(window_counts.max().item()) if B else 0
    Tmax = 0
    for bi in range(B):
        wc = int(window_counts[bi].item())
        for wi in range(wc):
            start = int(windows[bi, wi, 0].item())
            end = int(windows[bi, wi, 1].item())
            if start < 0 or end < start or (max_frames is not None and end >= max_frames):
                suffix = "" if max_frames is None else f" for frame tensor with F={max_frames}"
                raise ValueError(f"invalid window [{start},{end}]{suffix}")
            Tmax = max(Tmax, end - start + 1)

    valid = torch.zeros(B, Wmax, Tmax, device=windows.device, dtype=torch.bool)
    b = torch.full((B, Wmax), -1, device=windows.device, dtype=torch.long)
    frame_index = torch.full(
        (B, Wmax, Tmax), -1, device=windows.device, dtype=torch.long
    )
    for bi in range(B):
        wc = int(window_counts[bi].item())
        for wi in range(wc):
            start = int(windows[bi, wi, 0].item())
            end = int(windows[bi, wi, 1].item())
            length = end - start + 1
            valid[bi, wi, :length] = True
            b[bi, wi] = length - 1
            frame_index[bi, wi, :length] = torch.arange(
                start, end + 1, device=windows.device
            )
    return valid, b, frame_index


def rearrange_frames_by_window_timestep(
    frame_tensor: torch.Tensor,
    windows: torch.Tensor,
    window_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a per-frame tensor into the document's (window, timestep) layout.

    Args:
        frame_tensor: [B,F,...] tensor whose first temporal axis is video frame.
        windows: [B,Wmax,2] inclusive [start,end] indices into frame_tensor
            (section-local when videos are split).
        window_counts: [B], number of real token windows for each example.

    Returns:
        packed: [B,Wmax,Tmax,...]
        valid: [B,Wmax,Tmax] boolean mask for non-padding timesteps
        b: [B,Wmax] final valid relative timestep (length-1), or -1 if padded
        frame_index: [B,Wmax,Tmax] index into frame_tensor, -1 if padded

    Frame indices become (window i, relative timestep t) coordinates. Confidence
    training packs scalar logits here and computes token targets separately,
    one window at a time.
    """
    if frame_tensor.ndim < 2:
        raise ValueError("frame_tensor must have shape [B,F,...]")
    B, Fmax = frame_tensor.shape[:2]
    valid, b, frame_index = build_window_layout(
        windows, window_counts, max_frames=Fmax
    )
    Wmax, Tmax = valid.shape[1:]

    trailing = frame_tensor.shape[2:]
    packed = frame_tensor.new_zeros((B, Wmax, Tmax) + trailing)

    for bi in range(B):
        wc = int(window_counts[bi].item())
        for wi in range(wc):
            start = int(windows[bi, wi, 0].item())
            end = int(windows[bi, wi, 1].item())
            if not (0 <= start <= end < Fmax):
                raise ValueError(
                    f"invalid window [{start},{end}] for frame tensor with F={Fmax}"
                )
            length = end - start + 1
            packed[bi, wi, :length] = frame_tensor[bi, start : end + 1]
    return packed, valid, b, frame_index


def frozen_next_token_logits_for_window(
    model: UnnobaModel,
    hv0: torch.Tensor,
    ht0: torch.Tensor,
    *,
    batch_index: int,
    window_index: int,
    start: int,
    end: int,
    video_length: int,
    text_available_at: Optional[torch.Tensor] = None,
    text_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return per-frame token logits for one teacher-forced aligned window."""
    m = model.mixed
    if model.cfg.text_fusion == "frame":
        if text_available_at is None or text_lengths is None:
            raise ValueError("Frame predictions require the full causal text availability")
        length = int(text_lengths[batch_index].item())
        frames = torch.arange(start, end + 1, device=hv0.device)
        allowed = text_available_at[batch_index, None, :length] <= frames[:, None]
        video = hv0[batch_index:batch_index + 1, start:end + 1]
        vcross, _ = m.video_cross(
            video, ht0[batch_index:batch_index + 1, :length], allowed=allowed[None],
        )
        return m.frame_logits(video, vcross)[0]
    Fmax = hv0.shape[1]
    length = end - start + 1
    frame_ids = torch.arange(Fmax, device=hv0.device)
    q0 = ht0[
        batch_index : batch_index + 1, window_index : window_index + 1
    ].expand(1, length, -1)
    cutoff = torch.arange(start, end + 1, device=hv0.device)
    allowed = frame_ids[None, None, :] <= cutoff[None, :, None]
    allowed = allowed & (frame_ids[None, None, :] < video_length)

    tcross, _ = m.text_cross(
        q0, hv0[batch_index : batch_index + 1], allowed=allowed
    )
    ht = m.fuse_text(q0, tcross)
    ht_hat = m.text_ffn_ln(ht + m.text_ffn(ht))
    return m.token_head(ht_hat)[0]


def frozen_next_token_logits_per_frame(
    model: UnnobaModel,
    x_video: torch.Tensor,
    x_text: torch.Tensor,
    batch: dict,
) -> torch.Tensor:
    """Evaluate the frozen token path at every frame belonging to a token window.

    Token-query variants use the teacher query for window i, with video visible
    through section-local frame start_i+t. Frame fusion instead queries the
    teacher text states available at each frame. In evaluation mode these match
    streaming with the same video context and commits at aligned boundaries;
    live confidence-based commits can produce different text prefixes.

    Callers control gradient tracking and freeze the model for confidence fitting.

    Returns token logits [B,F,V]. Frames outside supplied real-token windows are
    left at zero and are subsequently masked out by the window packer.
    """
    m = model.mixed
    hv0 = m.vproj(x_video)
    if model.cfg.text_fusion == "frame":
        batch = dict(batch, windows=batch["frame_windows"], window_counts=batch["frame_window_counts"])
    ht0 = m.project_text(x_text if model.cfg.text_fusion == "frame" else select_text_queries(x_text, batch))
    B, Fmax, _ = hv0.shape
    V = model.cfg.vocab_size
    frame_logits = hv0.new_zeros(B, Fmax, V)

    for bi in range(B):
        fv = int(batch["video_lengths"][bi].item())
        n_windows = int(batch_window_counts(batch)[bi].item())
        for wi in range(n_windows):
            start = int(batch["windows"][bi, wi, 0].item())
            end = int(batch["windows"][bi, wi, 1].item())
            start = max(0, min(fv - 1, start))
            end = max(start, min(fv - 1, end))
            logits = frozen_next_token_logits_for_window(
                model, hv0, ht0,
                batch_index=bi,
                window_index=wi,
                start=start,
                end=end,
                video_length=fv,
                text_available_at=batch.get("text_available_at"), text_lengths=batch["text_lengths"],
            )
            frame_logits[bi, start : end + 1] = logits

    return frame_logits


def document_confidence_targets(
    Y: torch.Tensor,
    valid: torch.Tensor,
    b: torch.Tensor,
    *,
    beta: float,
    entropy_lambda: float,
    future_weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Implement U, D, Q, S and s=1-S from the document.

    Y: [B,W,T,V] probabilities after token-temperature softmax, clamped and
        renormalized internally for stable logarithms.
    valid: [B,W,T] valid window timesteps.
    b: [B,W] final valid relative timestep for each window.

    The draft constrains w_u >= 0 and sum_u w_u = 1 but does not prescribe a
    particular sequence. The default here is uniform over the admissible future
    offsets u=0..b_i-t. Setting future_weight_decay>0 uses exp(-decay*u),
    renormalized over that same admissible set; both choices obey the draft.
    """
    if beta <= 0:
        raise ValueError("confidence beta must be > 0")
    if not (0.0 <= entropy_lambda <= 1.0):
        raise ValueError("confidence lambda must be in [0,1]")
    if future_weight_decay < 0:
        raise ValueError("future_weight_decay must be >= 0")

    Y = Y.float().clamp_min(eps)
    Y = Y / Y.sum(dim=-1, keepdim=True).clamp_min(eps)
    B, W, T, V = Y.shape
    future_block = 128 if torch.is_autocast_enabled(Y.device.type) else max(1, T)
    logV = math.log(float(V))
    log2 = math.log(2.0)

    # U_{i,t} = H(Y_{i,t}) / log V.
    logY = Y.log()
    U = -(Y * logY).sum(dim=-1) / logV
    Q = torch.zeros(B, W, T, device=Y.device, dtype=Y.dtype)

    # D_{i,t,u} is Jensen-Shannon divergence between the distribution at t and
    # the distribution t+u (including u=0). Q uses normalized offset weights,
    # floored at eps before taking their logarithms.
    for bi in range(B):
        for wi in range(W):
            last = int(b[bi, wi].item())
            if last < 0:
                continue
            for ti in range(last + 1):
                count = last - ti + 1
                p = Y[bi, wi, ti]                         # [V]
                log_p = logY[bi, wi, ti]
                divergences = []
                for start in range(ti, last + 1, future_block):
                    stop = min(last + 1, start + future_block)
                    future = Y[bi, wi, start:stop]
                    log_future = logY[bi, wi, start:stop]
                    m = 0.5 * (p.unsqueeze(0) + future)
                    log_m = m.log()
                    kl_p_m = (
                        p.unsqueeze(0) * (log_p.unsqueeze(0) - log_m)
                    ).sum(dim=-1)
                    kl_f_m = (future * (log_future - log_m)).sum(dim=-1)
                    divergences.append(0.5 * (kl_p_m + kl_f_m))
                    del m, log_m
                # Only scalar divergences coexist across chunks; vocabulary
                # temporaries are bounded while all probability math stays FP32.
                D = torch.cat(divergences)              # [count], in [0,log 2]
    
                u = torch.arange(count, device=Y.device, dtype=Y.dtype)
                if future_weight_decay == 0.0:
                    w = torch.ones_like(u)
                else:
                    w = torch.exp(-future_weight_decay * u)
                w = w / w.sum().clamp_min(eps)
                Q[bi, wi, ti] = torch.logsumexp(
                    torch.log(w.clamp_min(eps)) + beta * D, dim=0
                ) / beta

    # S = lambda U + (1-lambda) Q/log(2), then s = 1-S.
    Q_norm = Q / log2
    S = entropy_lambda * U + (1.0 - entropy_lambda) * Q_norm
    # The exact normalized quantities lie in [0,1]. Clamp overshoot from
    # floating-point arithmetic and the epsilon floor on log weights.
    S = S.clamp(0.0, 1.0)
    s = (1.0 - S).clamp(0.0, 1.0)

    # Keep padded positions inert for diagnostics and downstream BCE masking.
    U = torch.where(valid, U, torch.zeros_like(U))
    Q = torch.where(valid, Q, torch.zeros_like(Q))
    Q_norm = torch.where(valid, Q_norm, torch.zeros_like(Q_norm))
    S = torch.where(valid, S, torch.zeros_like(S))
    s = torch.where(valid, s, torch.zeros_like(s))
    return s, {"U": U, "Q": Q, "Q_norm": Q_norm, "S": S}


def windowed_document_confidence_targets(
    model: UnnobaModel,
    x_video: torch.Tensor,
    x_text: torch.Tensor,
    batch: dict,
    window_counts: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the document target one window at a time.

    This is mathematically the same operation as constructing ``Y[B,F,V]``,
    packing it as ``Y[B,W,T,V]`` and calling
    :func:`document_confidence_targets`. Each target depends only on distributions
    from its own window, so peak vocabulary storage falls from ``B*W*T*V`` to
    ``max_window_length*V`` without a dense allocation across all windows.
    """
    B, Fmax = x_video.shape[:2]
    valid, b, frame_index = build_window_layout(
        batch["windows"], window_counts, max_frames=Fmax
    )
    shape = valid.shape
    s = torch.zeros(shape, device=x_video.device, dtype=torch.float32)
    target_stats = {
        name: torch.zeros_like(s) for name in ("U", "Q", "Q_norm", "S")
    }

    m = model.mixed
    hv0 = m.vproj(x_video)
    ht0 = m.project_text(x_text if model.cfg.text_fusion == "frame" else select_text_queries(x_text, batch))
    tau = float(args.confidence_token_temperature)

    for bi in range(B):
        fv = int(batch["video_lengths"][bi].item())
        wc = int(window_counts[bi].item())
        for wi in range(wc):
            start = int(batch["windows"][bi, wi, 0].item())
            end = int(batch["windows"][bi, wi, 1].item())
            length = end - start + 1
            logits = frozen_next_token_logits_for_window(
                model, hv0, ht0,
                batch_index=bi,
                window_index=wi,
                start=start,
                end=end,
                video_length=fv,
                text_available_at=batch.get("text_available_at"), text_lengths=batch["text_lengths"],
            )
            Y = torch.softmax(logits.float() / tau, dim=-1)
            window_valid = torch.ones(
                1, 1, length, device=Y.device, dtype=torch.bool
            )
            window_b = torch.full(
                (1, 1), length - 1, device=Y.device, dtype=torch.long
            )
            window_s, window_stats = document_confidence_targets(
                Y[None, None, ...], window_valid, window_b,
                beta=float(args.confidence_beta),
                entropy_lambda=float(args.confidence_lambda),
                future_weight_decay=float(args.confidence_future_weight_decay),
            )
            s[bi, wi, :length] = window_s[0, 0]
            for name, values in window_stats.items():
                target_stats[name][bi, wi, :length] = values[0, 0]

            # Do not let the final window's vocabulary tensors survive into
            # the trainable confidence forward below.
            del logits, Y, window_s, window_stats

    target_stats.update({"valid": valid, "b": b, "frame_index": frame_index})
    return s, target_stats


def document_confidence_loss(
    model: UnnobaModel,
    batch: dict,
    args,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Confidence loss from the supplied document, including window packing.

    The next-token model is frozen. Its causal distributions are evaluated and
    reduced to the document target one window at a time. This is equivalent to
    rearranging a dense Y into [batch, window, timestep, vocabulary], without
    retaining that prohibitively large tensor. The resulting soft target
    s=1-S is detached; gradients flow only through the confidence/video branch.
    """
    if model.cfg.text_fusion == "frame":
        batch = dict(batch, windows=batch["frame_windows"], window_counts=batch["frame_window_counts"])
    window_counts = batch_window_counts(batch)

    # Frozen encoders and frozen token-prediction path generate Y. Keeping this
    # under no_grad both enforces the intended freeze and avoids a large graph.
    with torch.no_grad():
        x_video = model.video_encoder(batch["video"], batch["video_lengths"]).detach()
        x_text = model.text_encoder(batch["teacher"], batch["text_lengths"]).detach()
        s, target_stats = windowed_document_confidence_targets(
            model, x_video, x_text, batch, window_counts, args
        )
        s = s.detach()
        valid = target_stats["valid"]
        b = target_stats["b"]
        frame_index = target_stats["frame_index"]

    # Confidence C uses the same causal video/text prefix availability as live
    # inference, but this branch remains trainable. The encoders are detached.
    vv, v2t = model.build_confidence_masks(
        batch["video_lengths"], batch["text_lengths"],
        batch["text_available_at"], x_video.shape[1], x_text.shape[1]
    )
    frame_conf_logits = model.mixed.forward_confidence(x_video, x_text, vv, v2t)
    C_logits, valid_c, b_c, _ = rearrange_frames_by_window_timestep(
        frame_conf_logits, batch["windows"], window_counts
    )
    if not torch.equal(valid, valid_c) or not torch.equal(b, b_c):
        raise RuntimeError("token/confidence window packing produced inconsistent masks")

    C = torch.sigmoid(C_logits.float() / float(args.confidence_temperature))
    eps = 1e-7
    point_loss = -(
        s * torch.log(C.clamp_min(eps))
        + (1.0 - s) * torch.log((1.0 - C).clamp_min(eps))
    )
    loss = point_loss[valid].mean()

    stats = dict(target_stats)
    stats.update({
        "s": s,
        "C": torch.where(valid, C, torch.zeros_like(C)),
        "valid": valid,
        "b": b,
        "frame_index": frame_index,
    })
    return loss, stats


def token_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
    position_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted CE divided by the number of valid positions, not weight mass.

    Keep targets sparse and bound AMP's FP32 vocabulary intermediates. Phasing
    gives current targets half weight; the separately masked loss adds the rest.
    """
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    if position_weights is not None:
        position_weights = position_weights.reshape(-1)
    count = (targets != pad_id).sum()
    if not count:
        return logits[:0].sum()  # Differentiable zero without summing huge logits.
    chunked = torch.is_autocast_enabled(logits.device.type) and logits.shape[0] > 128
    if not chunked and position_weights is None:
        return F.cross_entropy(logits, targets, ignore_index=pad_id)

    def chunk_loss(values, labels, weights):
        if weights is None:
            return F.cross_entropy(values, labels, ignore_index=pad_id, reduction="sum")
        dtype = torch.float32 if values.dtype in (torch.float16, torch.bfloat16) else values.dtype
        log_probs = F.log_softmax(values, dim=-1, dtype=dtype)
        loss = F.nll_loss(log_probs, labels, ignore_index=pad_id, reduction="none")
        return (weights * loss).sum()

    if not chunked:
        return chunk_loss(logits, targets, position_weights) / count

    losses = []
    for start in range(0, logits.shape[0], 128):
        values, labels = logits[start:start + 128], targets[start:start + 128]
        weights = None if position_weights is None else position_weights[start:start + 128]
        if torch.is_grad_enabled() and values.requires_grad:
            # Otherwise autograd retains all chunks' FP32 log-softmax outputs.
            loss = checkpoint(chunk_loss, values, labels, weights, use_reentrant=False)
        else:
            loss = chunk_loss(values, labels, weights)
        losses.append(loss)
    return torch.stack(losses).sum() / count


def previous_frame_cross_entropy(
    model: UnnobaModel, batch: dict, out: dict, pad_id: int,
) -> torch.Tensor:
    """Sum previous-target CE using text strictly before the previous token.

    Only phased frames run the extra attention/head, reusing encoded features
    and projected Q/K/V. Checkpoint each small chunk through CE so backward
    never retains a second full frame-by-vocabulary tensor (even without AMP).
    """
    previous = batch["frame_previous_targets"]
    phased = (previous != pad_id) & (batch["frame_targets"] != pad_id)
    mixed = model.mixed

    def chunk_loss(video, q, k, v, allowed, labels):
        vcross, _ = mixed.video_cross._attend(q, k, v, allowed=allowed)
        logits = mixed.frame_logits(video, vcross).squeeze(0)
        dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
        return F.cross_entropy(logits.to(dtype), labels, reduction="sum")

    losses = []
    for bi in range(phased.shape[0]):
        frames = phased[bi].nonzero(as_tuple=True)[0]
        if not frames.numel():
            continue
        length = int(batch["text_lengths"][bi])
        keys = torch.arange(length, device=frames.device)
        available = batch["text_available_at"][bi, :length]
        k = out["frame_k"][bi:bi + 1, :, :length]
        v = out["frame_v"][bi:bi + 1, :, :length]
        for indices in frames.split(128):
            # In a phased frame the latest available teacher state contains
            # the previous target. Remove that state as well as all later
            # states, which could also encode it. BOS remains visible.
            prefix_length = (available[None, :] <= indices[:, None]).sum(-1) - 1
            allowed = (keys[None, :] < prefix_length[:, None]).unsqueeze(0)
            video = out["video_projected"][bi:bi + 1].index_select(1, indices)
            q = out["frame_q"][bi:bi + 1].index_select(2, indices)
            labels = previous[bi, indices]
            if torch.is_grad_enabled():
                loss = checkpoint(chunk_loss, video, q, k, v, allowed, labels, use_reentrant=False)
            else:
                loss = chunk_loss(video, q, k, v, allowed, labels)
            losses.append(loss)
    return torch.stack(losses).sum() if losses else out["token_logits"].reshape(-1)[:0].sum()


def compute_losses(model: UnnobaModel, batch: dict, out: dict, args) -> Dict[str, torch.Tensor]:
    is_frame = model.cfg.text_fusion == "frame"
    targets = batch["frame_targets"] if is_frame else batch["targets"]
    query_lengths = batch["text_lengths"] if is_frame else batch_query_lengths(batch)
    window_counts = batch_window_counts(batch)
    active = (targets != args.pad_id).any(dim=1)
    active_video_lengths = batch["video_lengths"] * active
    query_lengths = query_lengths * active
    amp_enabled = torch.is_autocast_enabled(out["token_logits"].device.type)
    previous_targets = batch.get("frame_previous_targets") if is_frame and args.window_phasing > 0 else None
    phased = None if previous_targets is None else (
        (previous_targets != args.pad_id) & (targets != args.pad_id)
    )
    if phased is not None and phased.any():
        weights = torch.where(phased, 0.5, 1.0)
        token = token_cross_entropy(out["token_logits"], targets, args.pad_id, weights)
        previous_loss = previous_frame_cross_entropy(model, batch, out, args.pad_id)
        token = token + 0.5 * previous_loss / (targets != args.pad_id).sum()
    else:
        token = token_cross_entropy(out["token_logits"], targets, args.pad_id)
    zero = token.new_zeros(())
    vproj = similarity_preservation_loss(
        out["video_encoded"], out["video_projected"], active_video_lengths
    ) if model.cfg.use_video_projection and (args.lambda_vproj != 0.0 or not amp_enabled) else zero
    tproj = similarity_preservation_loss(
        out["text_encoded"], out["text_projected"], query_lengths
    ) if args.lambda_tproj != 0.0 or not amp_enabled else zero
    vnorm = norm_preservation_loss(
        out["video_encoded"], out["video_projected"], active_video_lengths
    ) if model.cfg.use_video_projection and (args.lambda_vnorm != 0.0 or not amp_enabled) else zero
    tnorm = norm_preservation_loss(
        out["text_encoded"], out["text_projected"], query_lengths
    ) if args.lambda_tnorm != 0.0 or not amp_enabled else zero
    if is_frame:
        if args.lambda_tcross or args.lambda_mono or args.lambda_align:
            raise ValueError("Frame fusion has no tcross or text-to-video attention losses")
        mono = align = token.new_zeros(())
    elif amp_enabled and args.lambda_mono == 0.0 and args.lambda_align == 0.0:
        mono = align = zero
    elif "text_video_q" in out:
        mono, align = flash_attention_losses(
            out["text_video_q"], out["text_video_k"], batch["token_video_end"],
            query_lengths, batch["video_lengths"],
            batch["windows"] if args.lambda_align != 0.0 else None,
            scale=model.mixed.text_cross.scale,
            window_counts=window_counts,
        )
        if amp_enabled and args.lambda_mono == 0.0:
            mono = zero
    else:
        mono = monotonic_attention_loss(
            out["text_video_attn"], query_lengths, batch["video_lengths"],
            window_counts=window_counts,
        ) if args.lambda_mono != 0.0 or not amp_enabled else zero
        align = aligned_window_attention_loss(
            out["text_video_attn"], batch["windows"], query_lengths, batch["video_lengths"],
            window_counts=window_counts,
        ) if args.lambda_align != 0.0 or not amp_enabled else zero
    total = (
        token
        + args.lambda_vproj * vproj
        + args.lambda_tproj * tproj
        + args.lambda_vnorm * vnorm
        + args.lambda_tnorm * tnorm
        + args.lambda_mono * mono
        + args.lambda_align * align
    )
    losses = {
        "total": total,
        "token": token,
        "vproj": vproj,
        "tproj": tproj,
        "vnorm": vnorm,
        "tnorm": tnorm,
        "mono": mono,
        "align": align,
    }
    # Leave the baseline objective and logging untouched when disabled.
    if args.lambda_tcross != 0.0:
        tcross_loss = tcross_margin_loss(
            out["tcross"], query_lengths, args.tcross_margin,
        )
        losses["tcross"] = tcross_loss
        losses["total"] = total + args.lambda_tcross * tcross_loss
    return losses


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            dtype = v.dtype
            if k == "video" and dtype != torch.float64 and torch.is_autocast_enabled(device.type):
                dtype = torch.get_autocast_dtype(device.type)
            out[k] = v.to(device=device, dtype=dtype)
        else:
            out[k] = v
    return out


def prepare_text_encoder_for_amp(model: UnnobaModel, device: torch.device) -> None:
    """Use compact frozen weights or FP32 trainable weights, only under AMP."""
    if not torch.is_autocast_enabled(device.type) or model.cfg.text_encoder_type != "pretrained_lm":
        return
    parameters = list(model.text_encoder.parameters())
    if not parameters or any(p.dtype == torch.float64 for p in parameters):
        return
    if any(p.requires_grad for p in parameters):
        # This also handles fine-tuning a previously frozen half-precision
        # checkpoint: GradScaler cannot unscale FP16 parameter gradients.
        dtype = torch.float32
    elif all(p.dtype == torch.float32 for p in parameters):
        dtype = torch.get_autocast_dtype(device.type)
    else:
        # Preserve an imported frozen FP16/BF16 model's precision policy.
        return
    with torch.no_grad():
        for parameter in parameters:
            parameter.data = parameter.data.to(dtype=dtype)
    # Keep numerical buffers (e.g. rotary frequencies) in their original dtype.
    # Reconstruct the same parameter dtype when resuming or evaluating offline.
    model.text_encoder.model.config.dtype = dtype
    model.cfg.pretrained_lm_config = model.text_encoder.model.config.to_dict()


METRIC_COLUMNS = (
    "variant", "checkpoint", "sample_index", "video", "text", "unit", "count",
    "avg_nll", "avg_margin",
)


def classification_metric_sums(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, int]:
    """Sum -log(p_true) and p_true - max(p_other) over already-valid positions."""
    if targets.numel() == 0:
        return 0.0, 0.0, 0
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    log_probs = F.log_softmax(logits.to(dtype), dim=-1)
    true_logp = log_probs.gather(-1, targets[:, None]).squeeze(-1)
    top = log_probs.topk(2, dim=-1)
    other_logp = torch.where(top.indices[:, 0] == targets, top.values[:, 1], top.values[:, 0])
    margin = true_logp.exp() - other_logp.exp()
    return float(-true_logp.double().sum()), float(margin.double().sum()), targets.numel()


@torch.inference_mode()
def evaluate_samples(
    model: UnnobaModel, dataset: VideoTextWindowDataset, *, device: torch.device,
    batch_size: int = 2, workers: int = 0, variant: str = "", checkpoint: str = "",
    include_eos: bool = False, amp: bool = False,
) -> List[dict]:
    """One teacher-forced, unregularized metric row per original manifest row.

    Aggregate numerators and counts across sections, never section means.
    Samples without scored positions have count=0 and blank means. Empty
    transcripts still score one EOS when include_eos=True in token-query variants.
    """
    model.eval()
    is_frame = model.cfg.text_fusion == "frame"
    tokenizer = dataset.tokenizer
    totals = [[0.0, 0.0, 0] for _ in dataset.rows]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        collate_fn=make_collate(tokenizer.pad_id))
    for batch in loader:
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            batch = move_batch(batch, device)
            # SDPA avoids returning attention weights. Metrics use at least
            # FP32 probability math and FP64 sums.
            logits = model(batch, flash_mono=True, need_attention=False)["token_logits"]
        targets = batch["frame_targets"] if is_frame else batch["targets"]
        lengths = batch["video_lengths"] if is_frame else batch_query_lengths(batch)
        valid = torch.arange(targets.shape[1], device=device)[None] < lengths[:, None]
        valid &= targets != tokenizer.pad_id
        if not is_frame and not include_eos:
            valid &= targets != tokenizer.eos_id
        for b, sample_index in enumerate(batch["sample_indices"]):
            # Limit additional vocabulary storage for long frame-level outputs.
            nll_sum = margin_sum = 0.0
            count = 0
            for start in range(0, logits.shape[1], 256):
                mask = valid[b, start:start + 256]
                nll, margin, n = classification_metric_sums(
                    logits[b, start:start + 256][mask], targets[b, start:start + 256][mask],
                )
                nll_sum += nll
                margin_sum += margin
                count += n
            total = totals[sample_index]
            total[0] += nll_sum
            total[1] += margin_sum
            total[2] += count
        if amp and device.type == "cuda":
            del logits
    rows = []
    for index, (sample, (nll_sum, margin_sum, count)) in enumerate(zip(dataset.rows, totals)):
        rows.append({
            "variant": variant or model.cfg.text_fusion,
            "checkpoint": str(checkpoint), "sample_index": index,
            "video": sample["video"], "text": sample["text"],
            "unit": "frame" if is_frame else "token", "count": count,
            "avg_nll": nll_sum / count if count else "",
            "avg_margin": margin_sum / count if count else "",
        })
    return rows


def write_metrics_csv(path: str, rows: Sequence[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: str,
    model: UnnobaModel,
    tokenizer,
    args,
    epoch: int,
    *,
    stage: str = "token",
    confidence_epoch: int = 0,
    confidence_trained: bool = False,
    validation_selection: Optional[dict] = None,
) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "model_config": asdict(model.cfg),
        "training_args": vars(args).copy(),
        "tokenizer": tokenizer.state_dict(),
        "tokenizer_pending": getattr(args, "tokenizer_pending", False),
        "preprocess": {
            "mouth_size": model.cfg.mouth_size,
            "use_face_detector": not args.no_face_detector,
        },
        "epoch": epoch,
        "pretrain_epoch": getattr(args, "pretrain_epoch", 0),
        "confidence_epoch": confidence_epoch,
        "training_stage": stage,
        "sectioning": {
            "max_frames": args.max_frames,
            "text_context": "full_prefix",
            "eos_target": "video_end_only",
        },
        "text_encoder": {
            "type": model.cfg.text_encoder_type,
            "pretrained_lm": model.cfg.pretrained_lm_name_or_path,
            "frozen": bool(model.cfg.freeze_text_encoder),
        },
        "confidence": {
            "trained": bool(confidence_trained),
            "target": "document_entropy_future_instability",
            "beta": float(args.confidence_beta),
            "lambda": float(args.confidence_lambda),
            "future_weight_decay": float(args.confidence_future_weight_decay),
            "future_weights": "uniform" if float(args.confidence_future_weight_decay) == 0.0 else "exp_decay_renormalized",
            "tau_token": float(args.confidence_token_temperature),
            "tau_conf": float(args.confidence_temperature),
        },
    }
    if validation_selection:
        ckpt["validation_selection"] = validation_selection
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def validate_early_stopping_args(args) -> None:
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be >= 0 (0 disables stopping)")
    if not math.isfinite(args.early_stopping_min_delta) or args.early_stopping_min_delta < 0:
        raise ValueError("--early-stopping-min-delta must be finite and >= 0")


@dataclass
class ValidationMonitor:
    """Stop on validation plateaus; select the exact minimum joint loss separately."""

    patience: int
    min_delta: float
    best_score: float = math.inf
    best_validation_loss: float = math.inf
    bad_epochs: int = 0
    best: Optional[dict] = None
    history: List[dict] = field(default_factory=list)

    def update(self, epoch, training_loss, validation_loss) -> bool:
        if not all(math.isfinite(value) for value in (training_loss, validation_loss)):
            raise ValueError("Non-finite training/validation loss; refusing to select this checkpoint")
        score = training_loss / 2 + validation_loss / 2
        record = {"epoch": epoch, "training_loss": training_loss,
                  "validation_loss": validation_loss, "score": score}
        self.history.append(record)
        improved = score < self.best_score
        if improved:
            self.best_score, self.best = score, record
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.patience > 0 and self.bad_epochs >= self.patience

    def state_dict(self):
        return {"criterion": "mean_training_validation_loss", "best": self.best,
                "history": self.history, "patience": self.patience,
                "min_delta": self.min_delta, "stopped_early": self.should_stop}


@torch.no_grad()
def evaluate_loss(model, loader, device, args, *, stage="token") -> float:
    """Evaluate the same objective on fixed weights, weighted by supervised positions."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    total, count = 0.0, 0
    try:
        for batch in loader:
            if stage == "token":
                targets = batch["frame_targets"] if model.cfg.text_fusion == "frame" else batch["targets"]
                weight = int((targets != args.pad_id).sum())
                if not weight:
                    continue
            else:
                counts = batch["frame_window_counts"] if model.cfg.text_fusion == "frame" else batch_window_counts(batch)
                if not (counts > 0).any():
                    continue
            with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                batch = move_batch(batch, device)
                if stage == "token":
                    out = model(batch, flash_mono=args.flash_mono,
                                need_attention=bool(args.lambda_mono or args.lambda_align))
                    loss = compute_losses(model, batch, out, args)["total"]
                    del out
                else:
                    loss, stats = document_confidence_loss(model, batch, args)
                    weight = int(stats["valid"].sum())
                    del stats
            total += float(loss) * weight
            count += weight
            del loss, batch
    finally:
        for module, mode in modes:
            module.training = mode
    if not count:
        raise ValueError(f"No supervised positions were available for {stage} loss evaluation")
    return total / count


def restore_checkpoint_weights(model, path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state"], strict=True)


def evaluate_streaming_accuracy(model, dataset, device, args) -> float:
    """Corpus token edit accuracy from full-video, free-running causal decoding.

    No teacher prefixes, alignment windows, section resets, or padding enter this
    metric. Insertions, deletions and substitutions all count as errors. Clamp
    once after summing edits/reference lengths over the entire split.
    """
    options = accuracy_decode_args(args)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    edits = reference_count = 0
    try:
        for row in dataset.rows:
            capture = cv2.VideoCapture(row["video"])
            try:
                if not capture.isOpened():
                    raise RuntimeError(f"Could not open video: {row['video']}")
                cropper = FixedMouthCropper(dataset.mouth_size, dataset.use_face_detector)

                def frames():
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            return
                        yield cropper(frame)

                prediction, count = decode_stream(
                    model, dataset.tokenizer, frames(), device, options,
                    token_temperature=options.token_temperature,
                    conf_temperature=options.conf_temperature, amp=args.amp,
                )
                if count == 0:
                    raise RuntimeError(f"No frames decoded from video: {row['video']}")
            finally:
                capture.release()
            reference = dataset.tokenizer.encode(row["text"])
            edits += token_edit_distance(reference, prediction)
            reference_count += len(reference)
    finally:
        for module, mode in modes:
            module.training = mode
    if not reference_count:
        return float(edits == 0)
    return max(0.0, 1.0 - edits / reference_count)


def _confidence_modules(model: UnnobaModel) -> List[nn.Module]:
    m = model.mixed
    modules = [
        m.video_ln, m.video_self_norm, m.video_self,
        m.video_self_ln, m.video_ffn, m.video_ffn_ln, m.conf_head,
    ]
    # In frame fusion video_cross also affects token logits and must stay frozen.
    return modules if model.cfg.text_fusion == "frame" else [m.video_cross, *modules]


def configure_confidence_stage(model: UnnobaModel) -> List[nn.Parameter]:
    """Freeze next-token behavior and expose only confidence-path params."""
    for p in model.parameters():
        p.requires_grad_(False)
    params: List[nn.Parameter] = []
    for module in _confidence_modules(model):
        for p in module.parameters():
            p.requires_grad_(True)
            params.append(p)
    return params


def set_confidence_train_mode(model: UnnobaModel) -> None:
    # Frozen encoders/projections stay deterministic; dropout remains active only
    # in the trainable confidence branch.
    model.eval()
    for module in _confidence_modules(model):
        module.train()


def configure_token_stage(model: UnnobaModel) -> List[nn.Parameter]:
    """Enable model gradients except for a frozen imported LM."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if model.cfg.text_encoder_type == "pretrained_lm" and model.cfg.freeze_text_encoder:
        for parameter in model.text_encoder.parameters():
            parameter.requires_grad_(False)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def pretrain_visual_encoder(model, loader, tokenizer, device, args, *, token_epoch=0) -> None:
    """Optimize only the video encoder, then restore the caller's trainability."""
    parameters = list(model.parameters())
    trainability = [p.requires_grad for p in parameters]
    modes = [(module, module.training) for module in model.modules()]
    for parameter in parameters:
        parameter.requires_grad_(False)
    visual_params = list(model.video_encoder.parameters())
    for parameter in visual_params:
        parameter.requires_grad_(True)
    amp_enabled = args.amp and device.type == "cuda"
    opt = torch.optim.AdamW(visual_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    global_step = 0
    try:
        for _ in range(args.pretrain_epochs):
            epoch = args.pretrain_epoch + 1
            model.eval()
            model.video_encoder.train()
            sums = {key: 0.0 for key in ("total", "temporal", "augmentation", "variance", "covariance")}
            n = 0
            for batch in loader:
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    batch = move_batch(batch, device)
                    losses = visual_pretraining_losses(
                        model.video_encoder, batch, args.pretrain_adjacent_frames,
                        lambda_temporal=args.lambda_pretrain_temporal,
                        lambda_augmentation=args.lambda_pretrain_augmentation,
                        lambda_variance=args.lambda_pretrain_variance,
                        lambda_covariance=args.lambda_pretrain_covariance,
                        variance_floor=args.pretrain_variance_floor,
                        original_probability=args.pretrain_original_probability,
                    )
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(visual_params, args.grad_clip)
                scaler.step(opt)
                scaler.update()
                for key in sums:
                    sums[key] += float(losses[key].detach())
                n += 1
                global_step += 1
                if args.log_every and global_step % args.log_every == 0:
                    msg = " ".join(f"{key}={value / n:.4f}" for key, value in sums.items())
                    print(f"pretrain epoch={epoch} step={global_step} {msg}")
                losses = None  # Release the forward graphs before the next batch.
            if n == 0:
                raise ValueError("No videos were available for visual pretraining")
            args.pretrain_epoch = epoch
            msg = " ".join(f"{key}={value / n:.4f}" for key, value in sums.items())
            print(f"pretrain epoch={epoch} done {msg}")
            save_checkpoint(args.output, model, tokenizer, args, token_epoch, stage="pretrain")
            print(f"saved {args.output}")
    finally:
        for parameter, requires_grad in zip(parameters, trainability):
            parameter.grad = None
            parameter.requires_grad_(requires_grad)
        for module, mode in modes:
            module.training = mode


def validate_pretraining_args(args) -> None:
    """Shared validation for the trainer, downloader, and variant runner."""
    if args.pretrain_only and args.confidence_only:
        raise ValueError("--pretrain-only cannot be combined with --confidence-only")
    if args.pretrain_epochs <= 0:
        raise ValueError("--pretrain-epochs must be > 0")
    if args.pretrain_adjacent_frames < 2:
        raise ValueError("--pretrain-adjacent-frames must be >= 2")
    for name in ("lambda_pretrain_temporal", "lambda_pretrain_augmentation",
                 "lambda_pretrain_variance", "lambda_pretrain_covariance"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and >= 0")
    if not math.isfinite(args.pretrain_variance_floor) or args.pretrain_variance_floor <= 0:
        raise ValueError("--pretrain-variance-floor must be finite and > 0")
    if not math.isfinite(args.pretrain_original_probability) or not 0 <= args.pretrain_original_probability <= 1:
        raise ValueError("--pretrain-original-probability must be finite and in [0,1]")


def train(args) -> None:
    validate_early_stopping_args(args)
    validate_pretraining_args(args)
    accuracy_decode_args(args)
    if args.pretrain_only:
        args.pretrain_visual_encoder = True
    if not args.manifest and not args.pretrain_only:
        raise ValueError("--manifest is required unless --pretrain-only is used")
    if args.pretrain_manifest and not args.pretrain_visual_encoder:
        raise ValueError("--pretrain-manifest requires --pretrain-visual-encoder")
    if args.pretrain_visual_encoder and not args.pretrain_manifest:
        raise ValueError("--pretrain-visual-encoder requires --pretrain-manifest")
    if args.pretrain_visual_encoder and args.confidence_only:
        raise ValueError("--pretrain-visual-encoder cannot be combined with --confidence-only")
    metrics_path = None
    if args.metrics_csv:
        metrics_path = str(Path(args.output).with_suffix(".metrics.csv")) if args.metrics_csv == "auto" else args.metrics_csv
        if Path(metrics_path).resolve() in {Path(path).resolve() for path in
                                           (args.output, args.manifest) if path}:
            raise ValueError("--metrics-csv must differ from the checkpoint and manifest paths")
    if args.pretrain_manifest:
        protected = {Path(args.output).resolve()}
        if metrics_path:
            protected.add(Path(metrics_path).resolve())
        if Path(args.pretrain_manifest).resolve() in protected:
            raise ValueError("--output and --metrics-csv must differ from --pretrain-manifest")
    for option in ("lambda_tcross", "tcross_margin"):
        value = getattr(args, option)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{option.replace('_', '-')} must be finite and >= 0")
    if not math.isfinite(args.w_t):
        raise ValueError("--w-t must be finite")
    if args.vcross_weight is not None and not math.isfinite(args.vcross_weight):
        raise ValueError("--vcross-weight must be finite")
    if not math.isfinite(args.window_phasing) or not 0 <= args.window_phasing <= 1:
        raise ValueError("--window-phasing must be finite and in [0,1]")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")
    if args.epochs < 0 or args.confidence_epochs < 0:
        raise ValueError("--epochs and --confidence-epochs must be >= 0")
    if args.pretrain_only:
        args.epochs = args.confidence_epochs = 0
    if args.confidence_only:
        if not args.resume:
            raise ValueError("--confidence-only requires --resume")
        if args.confidence_epochs == 0:
            raise ValueError("--confidence-only requires --confidence-epochs > 0")
        args.epochs = 0
    pretrain_only = args.pretrain_visual_encoder and args.epochs == 0 and args.confidence_epochs == 0
    if args.epochs == 0 and not args.resume and not pretrain_only:
        raise ValueError("--epochs 0 requires --resume; otherwise the token model would be random")
    if args.confidence_beta <= 0:
        raise ValueError("--confidence-beta must be > 0")
    if not (0.0 <= args.confidence_lambda <= 1.0):
        raise ValueError("--confidence-lambda must be in [0,1]")
    if args.confidence_future_weight_decay < 0:
        raise ValueError("--confidence-future-weight-decay must be >= 0")
    if args.confidence_token_temperature <= 0 or args.confidence_temperature <= 0:
        raise ValueError("confidence temperatures must be > 0")

    # Pretraining has no supervised metrics or learning curves.
    curve_args = argparse.Namespace(**vars(args))
    if pretrain_only:
        curve_args.plot_learning_curves = False
    curves = LearningCurveLogger(curve_args)
    if curves.selected:
        protected = {Path(path).resolve() for path in
                     (args.output, args.manifest, args.pretrain_manifest,
                      args.validation_manifest, args.resume, metrics_path) if path}
        if protected & {curves.csv_path.resolve(), curves.plot_path.resolve()}:
            raise ValueError("Learning-curve output paths must differ from checkpoints, manifests and --metrics-csv")
        print(f"learning curves enabled: {', '.join(curves.selected)}; "
              "streaming accuracy evaluates complete videos after each epoch")

    seed_everything(args.seed)
    rows = load_manifest(args.manifest) if args.manifest else []
    pretrain_rows = load_manifest(args.pretrain_manifest, video_only=True) if args.pretrain_visual_encoder else None
    validation_rows = load_manifest(args.validation_manifest) if args.validation_manifest else None
    if validation_rows is not None:
        validate_validation_split(rows, validation_rows)
        if pretrain_rows is not None:
            validate_validation_split(pretrain_rows, validation_rows)
        if Path(args.output).resolve() == Path(args.validation_manifest).resolve():
            raise ValueError("--output must differ from --validation-manifest")
        if metrics_path and Path(metrics_path).resolve() == Path(args.validation_manifest).resolve():
            raise ValueError("--metrics-csv must differ from --validation-manifest")
    resume_ckpt = None
    initialize_vocabulary = False
    args.tokenizer_pending = False
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**resume_ckpt["model_config"])
        if args.text_fusion is not None and args.text_fusion != cfg.text_fusion:
            raise ValueError(
                f"--resume text fusion {cfg.text_fusion!r} does not match requested "
                f"{args.text_fusion!r}; use a matching variant checkpoint or train from scratch"
            )
        if args.no_vproj and cfg.use_video_projection:
            raise ValueError("--no-vproj cannot resume a checkpoint with video projection; train from scratch")
        if args.vcross_weight is not None and args.vcross_weight != cfg.frame_vcross_weight:
            raise ValueError(
                f"--resume vcross weight {cfg.frame_vcross_weight:g} does not match requested "
                f"{args.vcross_weight:g}; omit --vcross-weight to restore the saved weight"
            )
        tokenizer = tokenizer_from_state_dict(resume_ckpt["tokenizer"])
        if cfg.text_encoder_type == "learned":
            args.tokenizer_pending = bool(resume_ckpt.get("tokenizer_pending", False))
            if args.tokenizer_pending and not pretrain_only and args.epochs == 0:
                raise ValueError("Resuming pretraining without a character vocabulary requires --epochs > 0")
            if rows:
                manifest_tokenizer = CharTokenizer.build(row["text"] for row in rows)
                if args.tokenizer_pending:
                    # No transcripts were available during pretraining. Only
                    # vocabulary-dependent weights need fresh initialization.
                    tokenizer = manifest_tokenizer
                    cfg.vocab_size = tokenizer.vocab_size
                    initialize_vocabulary = True
                    args.tokenizer_pending = False
                elif tokenizer.state_dict().get("itos") != manifest_tokenizer.state_dict().get("itos"):
                    raise ValueError("--resume tokenizer does not match the manifest's character vocabulary")
            if args.pretrained_lm:
                raise ValueError("--pretrained-lm cannot be used when resuming a learned-text checkpoint")
            if args.fine_tune_pretrained_lm:
                raise ValueError("--fine-tune-pretrained-lm requires a pretrained-LM checkpoint")
        else:
            if args.pretrained_lm and args.pretrained_lm != cfg.pretrained_lm_name_or_path:
                raise ValueError(
                    f"--pretrained-lm {args.pretrained_lm!r} does not match checkpoint LM "
                    f"{cfg.pretrained_lm_name_or_path!r}"
                )
            if args.fine_tune_pretrained_lm:
                cfg.freeze_text_encoder = False
    else:
        hf_config_dict = None
        d_text = args.d_text
        if args.pretrained_lm:
            tokenizer = HuggingFaceTokenizer.from_pretrained(
                args.pretrained_lm,
                local_files_only=args.pretrained_lm_local_files_only,
            )
            _, AutoConfig, _, _, _ = _require_huggingface()
            hf_config = AutoConfig.from_pretrained(
                args.pretrained_lm,
                local_files_only=args.pretrained_lm_local_files_only,
            )
            d_text = int(getattr(hf_config, "hidden_size"))
            hf_config_dict = hf_config.to_dict()
        else:
            if args.fine_tune_pretrained_lm:
                raise ValueError("--fine-tune-pretrained-lm requires --pretrained-lm")
            tokenizer = CharTokenizer.build(row["text"] for row in rows)
            args.tokenizer_pending = not rows

        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            mouth_size=args.mouth_size,
            conv3d_channels=args.conv3d_channels,
            d_video=args.d_video,
            d_text=d_text,
            d_fusion=args.d_video if args.no_vproj else args.d_fusion,
            heads=args.heads,
            video_layers=args.video_layers,
            text_layers=args.text_layers,
            ff_mult=args.ff_mult,
            conv_kernel=args.conv_kernel,
            dropout=args.dropout,
            text_encoder_type="pretrained_lm" if args.pretrained_lm else "learned",
            pretrained_lm_name_or_path=args.pretrained_lm,
            pretrained_lm_config=hf_config_dict,
            freeze_text_encoder=bool(args.pretrained_lm and not args.fine_tune_pretrained_lm),
            pretrained_lm_local_files_only=args.pretrained_lm_local_files_only,
            text_fusion=args.text_fusion or "residual",
            text_residual_weight=args.w_t,
            use_video_projection=not args.no_vproj,
            frame_vcross_weight=1.0 if args.vcross_weight is None else args.vcross_weight,
        )
    if args.vcross_weight is not None and cfg.text_fusion != "frame":
        raise ValueError("--vcross-weight requires --text-fusion frame")
    if args.window_phasing > 0 and cfg.text_fusion != "frame":
        raise ValueError("--window-phasing requires --text-fusion frame")
    if cfg.text_fusion == "frame" and (args.lambda_tcross or args.lambda_mono or args.lambda_align):
        raise ValueError("Frame fusion requires --lambda-tcross, --lambda-mono and --lambda-align to be 0")
    args.pad_id = tokenizer.pad_id
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_enabled = args.amp and device.type == "cuda"
    video_dtype = torch.get_autocast_dtype(device.type) if amp_enabled else torch.float32
    ds = None
    if not pretrain_only:
        ds = VideoTextWindowDataset(
            rows, tokenizer, cfg.mouth_size, use_face_detector=not args.no_face_detector,
            max_frames=args.max_frames, video_dtype=video_dtype,
            window_phasing=args.window_phasing,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=make_collate(tokenizer.pad_id),
            pin_memory=torch.cuda.is_available(),
        )
    validation_selection = {}
    token_monitor = confidence_monitor = None
    validation_ds = None
    def loss_loader(dataset):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, collate_fn=make_collate(tokenizer.pad_id),
                          pin_memory=device.type == "cuda",
                          generator=torch.Generator().manual_seed(args.seed))
    if not pretrain_only and (validation_rows is not None or curves.selected):
        training_loss_loader = loss_loader(ds)
    if validation_rows is not None and not pretrain_only:
        validation_ds = VideoTextWindowDataset(
            validation_rows, tokenizer, cfg.mouth_size, use_face_detector=not args.no_face_detector,
            max_frames=args.max_frames, video_dtype=ds.video_dtype,
            window_phasing=args.window_phasing,
        )
        validation_loader = loss_loader(validation_ds)
        token_monitor = ValidationMonitor(args.early_stopping_patience, args.early_stopping_min_delta)
        confidence_monitor = ValidationMonitor(args.early_stopping_patience, args.early_stopping_min_delta)
        print(f"validation videos={len(validation_rows)}; select lowest (training_loss + validation_loss) / 2; "
              f"validation patience={args.early_stopping_patience}")

    def record_learning_epoch(stage, epoch, confidence_trained, training_loss=None, validation_loss=None):
        if not curves.selected:
            return
        values = dict(training_loss=training_loss, validation_loss=validation_loss)
        for split, dataset in (("training", ds), ("validation", validation_ds)):
            if dataset is None:
                continue
            loss_key, accuracy_key = f"{split}_loss", f"{split}_accuracy"
            if loss_key in curves.selected and values[loss_key] is None:
                evaluation_loader = training_loss_loader if split == "training" else validation_loader
                values[loss_key] = evaluate_loss(model, evaluation_loader, device, args, stage=stage)
            if accuracy_key in curves.selected:
                values[accuracy_key] = evaluate_streaming_accuracy(model, dataset, device, args)
        curves.record(stage, epoch, confidence_trained, **values)
        print("learning curves: " + " ".join(f"{key}={values[key]:.6f}" for key in curves.selected))

    model = UnnobaModel(
        cfg,
        initialize_pretrained_text_encoder=bool(
            cfg.text_encoder_type == "pretrained_lm" and resume_ckpt is None
        ),
    )

    start_epoch = 0
    args.pretrain_epoch = 0
    prior_confidence_epoch = 0
    confidence_already_trained = False
    if resume_ckpt is not None:
        if initialize_vocabulary:
            fresh_state = model.state_dict()
            for name in ("text_encoder.embed.weight", "mixed.token_head.weight", "mixed.token_head.bias"):
                resume_ckpt["model_state"][name] = fresh_state[name]
            del fresh_state
            print("Initialized character vocabulary from the supervised manifest")
        model.load_state_dict(resume_ckpt["model_state"], strict=True)
        start_epoch = int(resume_ckpt.get("epoch", 0) or 0)
        args.pretrain_epoch = int(resume_ckpt.get("pretrain_epoch", 0) or 0)
        prior_confidence_epoch = int(resume_ckpt.get("confidence_epoch", 0) or 0)
        confidence_already_trained = bool(resume_ckpt.get("confidence", {}).get("trained", False))
        if args.epochs == 0:
            validation_selection.update(resume_ckpt.get("validation_selection", {}))
            if args.confidence_epochs > 0:
                validation_selection.pop("confidence", None)
        print(f"resumed {args.resume} token_epoch={start_epoch} confidence_epoch={prior_confidence_epoch}")
        # load_state_dict has copied every tensor into the model. Keeping the
        # CPU checkpoint mapping alive would retain a second full set of model
        # weights throughout training.
        del resume_ckpt

    # Convert eligible frozen LM parameters before transferring them to CUDA,
    # avoiding an initial full-size FP32 device allocation as well.
    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        prepare_text_encoder_for_amp(model, device)
    model = model.to(device)

    # Any update to shared token/video features invalidates a previously fitted
    # confidence calibration, so token training or pretraining resets its status.
    if args.epochs > 0 or args.pretrain_visual_encoder:
        prior_confidence_epoch = 0
        confidence_already_trained = False
        if args.pretrain_visual_encoder:
            validation_selection.clear()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"device={device} examples={len(ds) if ds is not None else 0} vocab={tokenizer.vocab_size} "
        f"text_encoder={cfg.text_encoder_type} params={total_params:,} trainable={trainable_params:,}"
    )
    print("window semantics: full text prefix; section-local token targets; EOS only at the true video end")
    if cfg.text_fusion != "residual" or args.lambda_tcross != 0.0:
        print(
            f"text_fusion={cfg.text_fusion} lambda_tcross={args.lambda_tcross:g} "
            f"tcross_margin={args.tcross_margin:g}"
        )
    if not cfg.use_video_projection:
        print(f"video projection disabled: fusion/text projection width={cfg.d_video}; video projection losses skipped")
    if cfg.text_fusion == "frame":
        print(f"frame vcross_weight={cfg.frame_vcross_weight:g} window_phasing={args.window_phasing:g}")
    if args.flash_mono and cfg.text_fusion != "frame":
        print("flash-mono: streamed pre-dropout attention losses; token attention uses SDPA")

    if pretrain_rows is not None:
        pretrain_ds = PretrainVideoDataset(
            pretrain_rows, cfg.mouth_size, use_face_detector=not args.no_face_detector,
            max_frames=args.max_frames, video_dtype=video_dtype,
        )
        pretrain_loader = DataLoader(
            pretrain_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
            collate_fn=collate_pretrain_videos, pin_memory=device.type == "cuda",
        )
        print(f"pretrain videos={len(pretrain_rows)} sections={len(pretrain_ds)} "
              f"adjacent_frames={args.pretrain_adjacent_frames}")
        pretrain_visual_encoder(model, pretrain_loader, tokenizer, device, args, token_epoch=start_epoch)
        del pretrain_loader, pretrain_ds
    if pretrain_only:
        print("Pretraining complete. Resume this checkpoint without pretraining flags to train token and confidence stages.")
        return

    # ------------------------------------------------------------------
    # Stage 1: token/frame prediction; confidence-only layers are not used.
    # ------------------------------------------------------------------
    if args.epochs > 0:
        token_params = configure_token_stage(model)
        opt = torch.optim.AdamW(token_params, lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
        global_step = 0
        final_token_epoch = start_epoch
        for local_epoch in range(1, args.epochs + 1):
            epoch = start_epoch + local_epoch
            final_token_epoch = epoch
            model.train()
            sums: Dict[str, float] = {}
            n = 0
            for batch in loader:
                targets = batch["frame_targets"] if cfg.text_fusion == "frame" else batch["targets"]
                if not (targets != args.pad_id).any():
                    continue
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    batch = move_batch(batch, device)
                    out = model(
                        batch, flash_mono=args.flash_mono,
                        need_attention=bool(args.lambda_mono or args.lambda_align),
                    )
                    losses = compute_losses(model, batch, out, args)
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(token_params, args.grad_clip)
                scaler.step(opt)
                scaler.update()

                for k, v in losses.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach().cpu())
                n += 1
                global_step += 1
                if args.log_every and global_step % args.log_every == 0:
                    keys = ["total", "token", "mono", "vproj", "tproj"]
                    if "tcross" in sums:
                        keys.append("tcross")
                    msg = " ".join(f"{k}={sums[k]/n:.4f}" for k in keys)
                    if cfg.text_fusion == "weighted":
                        msg += f" w_t={model.mixed.w_t.detach().item():.6f}"
                    print(f"token epoch={epoch} step={global_step} {msg}")

                if amp_enabled:
                    # Large logits/attention outputs must not overlap the next forward.
                    out = losses = None

            keys = ["total", "token", "mono", "vproj", "tproj", "vnorm", "tnorm", "align"]
            if "tcross" in sums:
                keys.append("tcross")
            msg = " ".join(f"{k}={sums[k]/max(1,n):.4f}" for k in keys)
            if cfg.text_fusion == "weighted":
                msg += f" w_t={model.mixed.w_t.detach().item():.6f}"
            print(f"token epoch={epoch} done {msg}")
            if n == 0:
                raise ValueError("No supervised positions were available for token/frame training")
            improved = True
            training_loss = validation_loss = None
            if token_monitor is not None or curves.selected:
                out = losses = None
                opt.zero_grad(set_to_none=True)
            if token_monitor is not None:
                training_loss = evaluate_loss(model, training_loss_loader, device, args)
                validation_loss = evaluate_loss(model, validation_loader, device, args)
                improved = token_monitor.update(epoch, training_loss, validation_loss)
                validation_selection["token"] = token_monitor.state_dict()
                print(f"token epoch={epoch} training_loss={training_loss:.6f} "
                      f"validation_loss={validation_loss:.6f} joint_loss={(training_loss + validation_loss) / 2:.6f}")
            record_learning_epoch("token", epoch, confidence_already_trained, training_loss, validation_loss)
            if improved:
                save_checkpoint(
                    args.output, model, tokenizer, args, epoch, stage="token",
                    confidence_epoch=prior_confidence_epoch,
                    confidence_trained=confidence_already_trained,
                    validation_selection=validation_selection,
                )
                print(f"saved {args.output}")
            if token_monitor is not None and token_monitor.should_stop:
                print(f"Early stopping token training at epoch {epoch}: validation loss stopped improving")
                break
    else:
        final_token_epoch = start_epoch

    # Stage one is finished. Its Adam moments and gradient buffers otherwise
    # remain referenced for the entire confidence stage even though none of
    # them can be reused there.
    if args.epochs > 0:
        for parameter in model.parameters():
            parameter.grad = None
        del opt, scaler, token_params, out, losses, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if token_monitor is not None:
            restore_checkpoint_weights(model, args.output)
            final_token_epoch = token_monitor.best["epoch"]
            save_checkpoint(args.output, model, tokenizer, args, final_token_epoch,
                            validation_selection=validation_selection)
            print(f"Restored best token checkpoint: epoch={final_token_epoch}")

    # ------------------------------------------------------------------
    # Stage 2: document confidence objective. Freeze every parameter that can
    # affect token logits; train only the document's video/confidence branch.
    # Targets are reduced window-by-window; scalar C is then rearranged to the
    # same (window,timestep) layout.
    # ------------------------------------------------------------------
    if args.confidence_epochs > 0:
        conf_params = configure_confidence_stage(model)
        conf_opt = torch.optim.AdamW(
            conf_params, lr=args.confidence_lr, weight_decay=args.weight_decay
        )
        conf_scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
        conf_global_step = 0
        for local_epoch in range(1, args.confidence_epochs + 1):
            conf_epoch = prior_confidence_epoch + local_epoch
            set_confidence_train_mode(model)
            total_loss = 0.0
            total_mean_target = 0.0
            n = 0
            for batch in loader:
                # Token-query variants use windows that finish in this section;
                # frame fusion also uses portions of unfinished token windows.
                # Neither silence nor EOS creates confidence targets.
                counts = batch["frame_window_counts"] if cfg.text_fusion == "frame" else batch_window_counts(batch)
                if not (counts > 0).any():
                    continue
                conf_opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                    batch = move_batch(batch, device)
                    conf_loss, conf_stats = document_confidence_loss(model, batch, args)
                conf_scaler.scale(conf_loss).backward()
                conf_scaler.unscale_(conf_opt)
                nn.utils.clip_grad_norm_(conf_params, args.grad_clip)
                conf_scaler.step(conf_opt)
                conf_scaler.update()

                valid = conf_stats["valid"]
                total_loss += float(conf_loss.detach().cpu())
                total_mean_target += float(conf_stats["s"][valid].mean().detach().cpu())
                mean_u = float(conf_stats["U"][valid].mean().detach().cpu())
                mean_q = float(conf_stats["Q_norm"][valid].mean().detach().cpu())
                mean_c = float(conf_stats["C"][valid].mean().detach().cpu())
                n += 1
                conf_global_step += 1
                if args.log_every and conf_global_step % args.log_every == 0:
                    print(
                        f"confidence epoch={conf_epoch} step={conf_global_step} "
                        f"loss={total_loss/n:.4f} mean_s={total_mean_target/n:.4f} "
                        f"U={mean_u:.4f} Q/log2={mean_q:.4f} C={mean_c:.4f}"
                    )

            print(
                f"confidence epoch={conf_epoch} done "
                f"loss={total_loss/max(1,n):.4f} mean_s={total_mean_target/max(1,n):.4f}"
            )
            improved = True
            training_loss = validation_loss = None
            if confidence_monitor is not None or curves.selected:
                if n == 0:
                    raise ValueError("No supervised positions were available for confidence training")
                conf_loss = conf_stats = None
                conf_opt.zero_grad(set_to_none=True)
            if confidence_monitor is not None:
                training_loss = evaluate_loss(model, training_loss_loader, device, args, stage="confidence")
                validation_loss = evaluate_loss(model, validation_loader, device, args, stage="confidence")
                improved = confidence_monitor.update(conf_epoch, training_loss, validation_loss)
                validation_selection["confidence"] = confidence_monitor.state_dict()
                print(f"confidence epoch={conf_epoch} training_loss={training_loss:.6f} "
                      f"validation_loss={validation_loss:.6f} joint_loss={(training_loss + validation_loss) / 2:.6f}")
            record_learning_epoch("confidence", conf_epoch, confidence_already_trained or n > 0,
                                  training_loss, validation_loss)
            if improved:
                save_checkpoint(
                    args.output, model, tokenizer, args, final_token_epoch,
                    stage="confidence", confidence_epoch=conf_epoch,
                    confidence_trained=confidence_already_trained or n > 0,
                    validation_selection=validation_selection,
                )
                print(f"saved {args.output}")
            if confidence_monitor is not None and confidence_monitor.should_stop:
                print(f"Early stopping confidence training at epoch {conf_epoch}: validation loss stopped improving")
                break
        if confidence_monitor is not None:
            restore_checkpoint_weights(model, args.output)
            save_checkpoint(args.output, model, tokenizer, args, final_token_epoch,
                            stage="confidence", confidence_epoch=confidence_monitor.best["epoch"],
                            confidence_trained=True, validation_selection=validation_selection)
            print(f"Restored best confidence checkpoint: epoch={confidence_monitor.best['epoch']}")


    if curves.history:
        print(f"saved learning curves: {curves.plot_path}; epoch metrics: {curves.csv_path}")
    if metrics_path is not None:
        rows = evaluate_samples(
            model, ds, device=device, batch_size=args.batch_size, workers=args.workers,
            variant=args.variant_name or Path(args.output).stem, checkpoint=args.output,
            include_eos=args.metrics_include_eos, amp=args.amp,
        )
        write_metrics_csv(metrics_path, rows)
        print(f"saved per-sample metrics: {metrics_path} ({len(rows)} rows)")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the causal Unnoba video-to-text prototype")
    p.add_argument(
        "--manifest", default=None,
        help="JSONL with video, text and per-character (or pretrained-token) frame windows; required unless --pretrain-only",
    )
    p.add_argument("--output", default="unnoba.pt")
    p.add_argument("--pretrain-visual-encoder", action="store_true",
                   help="Pretrain the full video encoder with VICReg-style losses and optional temporal MSE")
    p.add_argument("--pretrain-only", action="store_true",
                   help="Run only visual pretraining and save a resumable checkpoint; enables --pretrain-visual-encoder, "
                        "overrides --epochs/--confidence-epochs to 0, and skips supervised metrics")
    p.add_argument("--pretrain-manifest", default=None,
                   help="Pretraining JSONL with video paths; text/windows are optional and ignored")
    p.add_argument("--pretrain-epochs", type=int, default=5,
                   help="Additional visual pretraining epochs when enabled (default: 5); uses --lr and --weight-decay")
    p.add_argument("--pretrain-adjacent-frames", type=int, default=2, metavar="N",
                   help="Compare each valid frame pair with 0 < j-i < N once (default: 2, consecutive pairs)")
    p.add_argument("--pretrain-original-probability", type=float, default=0.2,
                   help="Probability of keeping each frame unmodified, independently per augmented view (default: 0.2; range [0,1])")
    p.add_argument("--lambda-pretrain-temporal", type=float, default=0.0,
                   help="Pretraining temporal MSE weight (default: 0; disabled)")
    p.add_argument("--lambda-pretrain-augmentation", type=float, default=1.0,
                   help="Pretraining augmented-view MSE weight (default: 1)")
    p.add_argument("--lambda-pretrain-variance", type=float, default=1.0,
                   help="Pretraining variance-floor weight, averaged over both views (default: 1)")
    p.add_argument("--lambda-pretrain-covariance", type=float, default=0.04,
                   help="Pretraining off-diagonal covariance weight, summed over both views (default: 0.04)")
    p.add_argument("--pretrain-variance-floor", type=float, default=1.0,
                   help="Target minimum per-feature standard deviation across valid frames (default: 1; must be > 0)")
    p.add_argument("--validation-manifest", default=None,
                   help="Separate validation JSONL; enables best-checkpoint selection and early stopping")
    p.add_argument("--early-stopping-patience", type=int, default=5,
                   help="Epochs without validation-loss improvement before stopping each stage (default: 5; 0 disables)")
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                   help="Minimum validation-loss decrease to reset patience (default: 0)")
    p.add_argument("--metrics-csv", default=None, metavar="PATH_OR_AUTO",
                   help="Write per-sample NLL/margin after training; auto uses CHECKPOINT_STEM.metrics.csv")
    p.add_argument("--metrics-include-eos", action="store_true",
                   help="Include EOS in token-level CSV means (default: transcript tokens only)")
    p.add_argument("--variant-name", default=None, help="Experiment label in the metrics CSV")
    add_learning_curve_arguments(p)
    p.add_argument("--resume", default=None,
                   help="Optional compatible checkpoint; pretraining-only checkpoints proceed to full training unless "
                        "pretraining flags are supplied again. Add --confidence-only to skip token training.")
    p.add_argument(
        "--confidence-only", action="store_true",
        help="Train only the frozen-token confidence stage; requires --resume and overrides --epochs to 0",
    )
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--max-frames", type=int, default=None, metavar="N",
        help="Split videos into consecutive sections of at most N frames per batch element (default: full videos)",
    )
    p.add_argument("--workers", type=int, default=0, help="Keep 0 if OpenCV decoding behaves badly in worker processes")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", action="store_true",
                   help="CUDA mixed precision with compact activations, checkpointed visual features, "
                        "and skipped zero-weight losses; also applies to final metrics")
    p.add_argument("--log-every", type=int, default=10)

    p.add_argument(
        "--pretrained-lm", default=None, metavar="MODEL_OR_PATH",
        help=(
            "Use this Hugging Face causal LM and its fast tokenizer as the text encoder. "
            "Its weights are frozen unless --fine-tune-pretrained-lm is set."
        ),
    )
    p.add_argument(
        "--fine-tune-pretrained-lm", action="store_true",
        help="Update imported LM weights during token training (default: keep them frozen)",
    )
    p.add_argument(
        "--pretrained-lm-local-files-only", action="store_true",
        help="Resolve --pretrained-lm only from local Hugging Face cache/files",
    )

    p.add_argument("--mouth-size", type=int, default=96)
    p.add_argument("--no-face-detector", action="store_true", help="Use deterministic lower-center crop instead of Haar face initialization")
    p.add_argument("--conv3d-channels", type=int, default=32)
    p.add_argument("--d-video", type=int, default=256)
    p.add_argument("--d-text", type=int, default=256)
    p.add_argument("--d-fusion", type=int, default=256,
                   help="Fusion feature width (ignored with --no-vproj, which uses --d-video)")
    p.add_argument("--no-vproj", action="store_true",
                   help="Use video features directly and project text to --d-video; resume restores the saved architecture")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--video-layers", type=int, default=2)
    p.add_argument("--text-layers", type=int, default=2)
    p.add_argument("--ff-mult", type=int, default=1)
    p.add_argument("--conv-kernel", type=int, default=7)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--text-fusion", choices=("residual", "cross_only", "weighted", "frame"), default=None,
        help="ht fusion: ht0+tcross, tcross, w_t*ht0+tcross, or per-frame hv0+vcross_weight*vcross; resume requires matching fusion",
    )
    p.add_argument("--w-t", type=float, default=1.0,
                   help="Fixed scalar w_t for weighted fusion (default: 1); resume restores the saved weight")
    p.add_argument("--vcross-weight", type=float, default=None,
                   help="Fixed vcross multiplier for frame token fusion (default: 1; 0 = video-only token prediction); resume restores the saved weight")
    p.add_argument("--window-phasing", type=float, default=0.0, metavar="FRACTION",
                   help="Frame fusion only: average current/previous token CE, each with text strictly before its target, on the first floor(FRACTION * window_length) frames; in [0,1], default 0 (disabled)")

    # Projection and attention regularizers.
    p.add_argument("--lambda-vproj", type=float, default=0.0)
    p.add_argument("--lambda-tproj", type=float, default=0.0)
    p.add_argument("--lambda-vnorm", type=float, default=0.01)
    p.add_argument("--lambda-tnorm", type=float, default=0.01)
    p.add_argument("--lambda-mono", type=float, default=0.0)
    p.add_argument(
        "--flash-mono", action="store_true",
        help=(
            "Stream monotonic and enabled alignment losses from Q/K with recomputing backward; "
            "use pre-dropout probabilities and SDPA for token attention (default: dense weights)"
        ),
    )
    p.add_argument("--lambda-align", type=float, default=0.0)
    p.add_argument("--lambda-tcross", type=float, default=0.0,
                   help="Weight of ReLU(m - ||tcross||_2)^2; 0 disables the penalty")
    p.add_argument("--tcross-margin", type=float, default=1.0, metavar="M",
                   help="Nonnegative threshold m for the tcross norm penalty")

    # Confidence-stage optimization and target parameters.
    p.add_argument("--confidence-epochs", type=int, default=5)
    p.add_argument("--confidence-lr", type=float, default=3e-4)
    p.add_argument("--confidence-beta", type=float, default=10.0,
                   help="beta in Q=(1/beta) log sum_u w_u exp(beta D)")
    p.add_argument("--confidence-lambda", type=float, default=0.5,
                   help="lambda mixing normalized entropy U and Q/log(2)")
    p.add_argument("--confidence-future-weight-decay", type=float, default=0.0,
                   help="0 = uniform w_u; >0 uses exp(-decay*u), renormalized for each valid future range")
    p.add_argument("--confidence-token-temperature", type=float, default=1.0,
                   help="tau_token used to form Y=softmax(token_logits/tau_token)")
    p.add_argument("--confidence-temperature", type=float, default=1.0,
                   help="tau_conf used to form C=sigmoid(conf_logits/tau_conf)")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
