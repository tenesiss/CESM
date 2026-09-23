#!/usr/bin/env python3
"""
Incremental / streaming-style inference for the Unnoba prototype.

The input is a normal video file, but frames are decoded and processed one at a
time. No future frame is read by the neural network before producing the current
next-token distribution.

Caches used during inference:
  * causal Conv3D input-frame history
  * video Conformer self-attention K/V per layer
  * video causal-convolution history per layer
  * text encoder self-attention K/V per layer
  * Mixed Block text->video cross-attention K/V (non-frame variants only)
  * Mixed Block video->text cross-attention K/V (grows per committed token)
  * Mixed Block video-path self-attention K/V

Token commitment is driven by the learned per-frame confidence head. The base
confidence threshold can relax as more frames pass since the previous commit.
Relaxation can reduce delays, but candidates must still reach the threshold floor
and satisfy the warmup, frame-gap and optional probability/stability guards.
Repeated token IDs must also wait a configurable number of frames since their
own last emission. Predictions during this cooldown are skipped.
While the same next-token candidate remains active, its peak confidence is kept;
this lets a later relaxed threshold accept an earlier peak. Confidence is trained
from token entropy and future-distribution instability, not explicit boundaries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from train import FixedMouthCropper, ModelConfig, UnnobaModel, tokenizer_from_state_dict
from streaming import add_decode_arguments, decode_stream, mask_invalid_generation_logits, validate_decode_arguments


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    tokenizer = tokenizer_from_state_dict(ckpt["tokenizer"])
    # Pretrained LM weights are already embedded in the CESM checkpoint. Build
    # its architecture from the stored HF config rather than downloading it.
    model = UnnobaModel(cfg, initialize_pretrained_text_encoder=False).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    preprocess = ckpt.get("preprocess", {})
    return model, tokenizer, preprocess, ckpt


def emit_event(args, event: dict) -> None:
    if args.json_events:
        print(json.dumps(event, ensure_ascii=False), flush=True)
    else:
        token = event.get("token", "")
        if token:
            sys.stdout.write(token)
            sys.stdout.flush()


def main(args) -> None:
    try:
        validate_decode_arguments(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, tokenizer, preprocess, ckpt = load_model(args.checkpoint, device)
    # Match the arithmetic used to train the projections and token head. A
    # stored BF16 text encoder may have run under FP16 CUDA autocast during
    # training; its parameter dtype alone does not determine that precision.
    amp_requested = args.amp if args.amp is not None else bool(ckpt.get("training_args", {}).get("amp", False))
    amp_enabled = bool(amp_requested and device.type == "cuda")
    confidence_meta = ckpt.get("confidence", {})
    # Unless explicitly overridden, use the same temperatures that defined Y
    # and C during confidence training. Older checkpoints fall back to 1.0.
    token_temperature = (
        float(args.token_temperature) if args.token_temperature is not None
        else float(confidence_meta.get("tau_token", 1.0))
    )
    conf_temperature = (
        float(args.conf_temperature) if args.conf_temperature is not None
        else float(confidence_meta.get("tau_conf", 1.0))
    )
    if token_temperature <= 0 or conf_temperature <= 0:
        raise SystemExit("token/confidence temperatures must be > 0")
    if not bool(confidence_meta.get("trained", False)):
        print(
            "warning: checkpoint is not marked as having a trained confidence head; "
            "commit timing may be meaningless. Train confidence with train.py "
            "(--confidence-epochs > 0), or resume an old checkpoint with "
            "--epochs 0 --resume <checkpoint>.",
            file=sys.stderr,
            flush=True,
        )
    elif confidence_meta.get("target") != "document_entropy_future_instability":
        print(
            "warning: this checkpoint's confidence head was trained with an older "
            "target rather than the document's entropy/future-instability objective. "
            "Resume it with train.py --epochs 0 --confidence-epochs ... to refit confidence.",
            file=sys.stderr,
            flush=True,
        )
    mouth_size = int(preprocess.get("mouth_size", model.cfg.mouth_size))
    use_face_detector = bool(preprocess.get("use_face_detector", True))
    if args.no_face_detector:
        use_face_detector = False

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    cropper = FixedMouthCropper(mouth_size, use_face_detector=use_face_detector)
    started_wall = time.perf_counter()

    if args.json_events:
        print(json.dumps({
            "event": "start",
            "video": str(Path(args.video)),
            "fps": fps,
            "device": str(device),
            "amp": amp_enabled,
            "amp_dtype": str(torch.get_autocast_dtype(device.type)) if amp_enabled else None,
            "checkpoint_epoch": ckpt.get("epoch"),
            "confidence_epoch": ckpt.get("confidence_epoch"),
            "confidence_trained": bool(confidence_meta.get("trained", False)),
            "confidence_threshold": args.confidence_threshold,
            "confidence_min_threshold": args.confidence_min_threshold,
            "confidence_relax_per_frame": args.confidence_relax_per_frame,
            "repeat_token_cooldown_frames": args.repeat_token_cooldown_frames,
            "token_temperature": token_temperature,
            "confidence_temperature": conf_temperature,
        }), flush=True)

    def frames():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                return
            yield cropper(frame_bgr)

    try:
        emitted_ids, frame_count = decode_stream(
            model, tokenizer, frames(), device, args,
            token_temperature=token_temperature, conf_temperature=conf_temperature,
            amp=amp_enabled, fps=fps, on_event=lambda event: emit_event(args, event),
            debug_every=args.debug_every, pace_realtime=args.pace_realtime,
        )
    finally:
        cap.release()
    frame_idx = frame_count - 1

    transcript = tokenizer.decode(emitted_ids)
    elapsed = time.perf_counter() - started_wall
    if args.json_events:
        print(json.dumps({
            "event": "done",
            "text": transcript,
            "frames": frame_idx + 1,
            "video_seconds": (frame_idx + 1) / fps if frame_idx >= 0 else 0.0,
            "wall_seconds": elapsed,
            "realtime_factor": elapsed / max((frame_idx + 1) / fps, 1e-9),
        }, ensure_ascii=False), flush=True)
    else:
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(
            f"processed {frame_idx + 1} frames ({(frame_idx + 1)/fps:.2f}s video) "
            f"in {elapsed:.2f}s; transcript={transcript!r}",
            file=sys.stderr,
        )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Causal cached inference for the Unnoba prototype")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="Override CUDA mixed precision (default: restore the checkpoint training setting; --no-amp disables it)")
    p.add_argument("--no-face-detector", action="store_true")

    add_decode_arguments(p)

    p.add_argument("--pace-realtime", action="store_true",
                   help="Sleep when processing is faster than the source FPS; otherwise run as fast as possible")
    p.add_argument("--json-events", action="store_true",
                   help="Emit one JSON object per token instead of a live plain-text stream")
    p.add_argument("--debug-every", type=int, default=0,
                   help="Print current next-token/attention diagnostics to stderr every N frames")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
