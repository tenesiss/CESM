#!/usr/bin/env python3
"""
Group video frames by aligned tokenizer tokens.

Given:
  - a video containing speech, or a folder of videos
  - a Hugging Face fast tokenizer

Produces:
  - one folder per token occurrence
  - the video frames assigned to that token
  - transcript.txt
  - words.json
  - manifest.json
  - data.jsonl with video, transcript text, and frame-group windows (CLI)
  - for folder input, one output tree per video plus batch_manifest.json

Important assignment rule:
  A frame is written to at most one token folder. Overlapping token timestamps
  move the group boundary to the frame containing the later token's start.
  Otherwise, the next group begins after the previous token ends, absorbing any
  intervening silence. Boundaries never move backward; equal starts favor the
  later token. Frames after the final spoken token are omitted.

Token timestamps are approximate at sub-word level. faster-whisper provides
word-level timestamps; when a word is split into several tokenizer tokens, this
script interpolates timestamps within that word using tokenizer character
offsets.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
from faster_whisper import WhisperModel
from transformers import AutoTokenizer


VIDEO_EXTENSIONS = frozenset(
    {
        ".3gp",
        ".avi",
        ".flv",
        ".m2ts",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".ogv",
        ".ts",
        ".webm",
        ".wmv",
    }
)
JSONL_FILENAME = "data.jsonl"


@dataclass
class WordSpan:
    text: str
    char_start: int
    char_end: int
    time_start: float
    time_end: float

    @property
    def content_char_start(self) -> int:
        # faster-whisper words often include leading whitespace.
        return self.char_start + (len(self.text) - len(self.text.lstrip()))

    @property
    def content_char_end(self) -> int:
        return self.char_end - (len(self.text) - len(self.text.rstrip()))


@dataclass
class TokenSpan:
    index: int
    token_id: int
    token: str
    text: str
    char_start: int
    char_end: int
    time_start: float
    time_end: float

    frame_count: int = 0
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    folder: str = ""


def transcribe_words(
    video_path: str | Path,
    *,
    whisper_model: str = "small",
    language: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "default",
    whisper_instance: Optional[WhisperModel] = None,
) -> tuple[str, list[WordSpan]]:
    """
    Transcribe video audio and reconstruct the transcript from timestamped words.
    """
    model = whisper_instance
    if model is None:
        model = WhisperModel(
            whisper_model,
            device=device,
            compute_type=compute_type,
        )

    segments, _info = model.transcribe(
        str(video_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
    )

    pieces: list[str] = []
    words: list[WordSpan] = []
    char_pos = 0

    # segments is a generator; iterating it performs the transcription.
    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            if word.start is None or word.end is None:
                continue

            piece = word.word or ""
            if not piece:
                continue

            char_start = char_pos
            pieces.append(piece)
            char_pos += len(piece)

            words.append(
                WordSpan(
                    text=piece,
                    char_start=char_start,
                    char_end=char_pos,
                    time_start=float(word.start),
                    time_end=float(word.end),
                )
            )

    return "".join(pieces), words


def _char_boundary_to_time(
    pos: int,
    words: list[WordSpan],
    *,
    side: str,
) -> Optional[float]:
    """
    Map a transcript character boundary to an approximate audio timestamp.

    For characters inside a spoken word, linearly interpolate through that
    word's time range. Leading/trailing whitespace maps to the word boundary.
    """
    if not words:
        return None

    if side not in {"start", "end"}:
        raise ValueError("side must be 'start' or 'end'")

    if pos <= words[0].char_start:
        return words[0].time_start
    if pos >= words[-1].char_end:
        return words[-1].time_end

    candidate: Optional[WordSpan] = None

    if side == "start":
        for word in words:
            if word.char_start <= pos < word.char_end:
                candidate = word
                break
            if pos < word.char_start:
                candidate = word
                break
    else:
        for word in words:
            if word.char_start < pos <= word.char_end:
                candidate = word
                break
            if pos <= word.char_start:
                break

        if candidate is None:
            for word in reversed(words):
                if word.char_end <= pos:
                    candidate = word
                    break

    if candidate is None:
        return None

    word = candidate
    content_start = word.content_char_start
    content_end = word.content_char_end

    # Defensive fallback for an all-whitespace ASR piece.
    if content_end <= content_start:
        return word.time_start if side == "start" else word.time_end

    if pos <= content_start:
        return word.time_start
    if pos >= content_end:
        return word.time_end

    fraction = (pos - content_start) / (content_end - content_start)
    return word.time_start + fraction * (word.time_end - word.time_start)


def align_tokens(
    transcript: str,
    words: list[WordSpan],
    tokenizer: Any,
) -> list[TokenSpan]:
    """
    Tokenize the full transcript, then align each token's character offsets
    against the timestamped ASR words.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "A Hugging Face FAST tokenizer is required. "
            "Load it with AutoTokenizer.from_pretrained(..., use_fast=True)."
        )

    encoded = tokenizer(
        transcript,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    spans: list[TokenSpan] = []

    for token_id, offset in zip(input_ids, offsets):
        char_start, char_end = int(offset[0]), int(offset[1])

        # Zero-width offsets can occur for tokenizer artifacts/special tokens.
        if char_end <= char_start:
            continue

        time_start = _char_boundary_to_time(
            char_start, words, side="start"
        )
        time_end = _char_boundary_to_time(
            char_end, words, side="end"
        )

        if time_start is None or time_end is None:
            continue

        # Avoid negative-duration windows due to pathological timestamp noise.
        if time_end < time_start:
            time_end = time_start

        spans.append(
            TokenSpan(
                index=len(spans),
                token_id=int(token_id),
                token=str(tokenizer.convert_ids_to_tokens(int(token_id))),
                text=transcript[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                time_start=float(time_start),
                time_end=float(time_end),
            )
        )

    return spans


def _safe_name(text: str, max_len: int = 48) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", "_").replace("\r", "_").replace("\t", "_")
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    return (text or "token")[:max_len]


def _build_group_starts(
    spans: list[TokenSpan],
    fps: float,
) -> list[int]:
    """
    Return the inclusive start frame for every token group.

    Group 0 starts at frame 0. At each later token boundary:
      * overlap -> the first frame touched by the NEXT token goes to it;
      * silence/gap -> the first frame after the previous token ends goes to
        the NEXT token.

    Starts may be equal when multiple tokens fall inside one source frame. In
    that case the later token wins that frame, and an earlier group can be
    empty. No source frame is duplicated between groups.
    """
    if not spans:
        return []

    starts = [0]

    for prev, cur in zip(spans, spans[1:]):
        if cur.time_start < prev.time_end:
            # The next token overlaps the previous one. The source frame that
            # first touches the next token belongs to the next token.
            raw_start = int(math.floor(cur.time_start * fps))
        else:
            # Non-overlapping timestamps include both gaps and touching windows.
            # Keep frames touched by the previous token in its group.
            raw_start = int(math.ceil(prev.time_end * fps))

        starts.append(max(starts[-1], raw_start, 0))

    return starts


def _build_inclusive_groups(
    spans: list[TokenSpan],
    fps: float,
    frames_decoded: int,
) -> list[list[int]]:
    """
    Build exactly this shape (inclusive source-frame indices):

        [[0, x0], [x0 + 1, x1], [x1 + 1, x2], ...]

    groups[i] corresponds to spans[i]. The final end is
    min(ceil(final_token_end * fps), frames_decoded) - 1, treating token end times
    as exclusive and omitting trailing video.

    An empty group is represented as [s, s-1]. This can happen when multiple
    tokenizer tokens occupy a single video frame.
    """
    if not spans or frames_decoded <= 0:
        return []

    starts = _build_group_starts(spans, fps)
    last_exclusive = min(
        max(int(math.ceil(spans[-1].time_end * fps)), 0),
        frames_decoded,
    )

    # Once the video or final spoken token ends, all later groups are empty at
    # that same boundary while preserving the x_(i-1)+1 adjacency invariant.
    starts = [min(s, last_exclusive) for s in starts]

    groups: list[list[int]] = []
    for i, group_start in enumerate(starts):
        next_start = starts[i + 1] if i + 1 < len(starts) else last_exclusive
        groups.append([int(group_start), int(next_start - 1)])

    return groups


def save_grouped_frames(
    video_path: str | Path,
    spans: list[TokenSpan],
    output_dir: str | Path,
    *,
    transcript: Optional[str] = None,
    image_ext: str = "jpg",
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    """
    Save frames into strictly non-overlapping, contiguous token groups.

    manifest["groups"] has one inclusive [start_frame, end_frame] pair per
    token occurrence and is indexed exactly like manifest["tokens"].
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_root = output_dir / "tokens"
    token_root.mkdir(parents=True, exist_ok=True)

    for span in spans:
        label = _safe_name(span.token)
        folder_name = f"{span.index:06d}_{label}"
        span.folder = str(Path("tokens") / folder_name)
        (output_dir / span.folder).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise RuntimeError("Video FPS is missing or invalid.")

    total_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ext = image_ext.lower().lstrip(".")
    if ext not in {"jpg", "jpeg", "png"}:
        cap.release()
        raise ValueError("--image-ext must be jpg, jpeg, or png")

    group_starts = _build_group_starts(spans, fps)
    last_speech_end_frame = int(math.ceil(spans[-1].time_end * fps)) - 1

    frame_index = 0
    group_index = 0
    assigned_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Advance through every token whose group has started. Using <= here
        # means equal start frames are resolved in favor of the later token.
        while (
            group_index + 1 < len(group_starts)
            and group_starts[group_index + 1] <= frame_index
        ):
            group_index += 1

        if spans and frame_index <= last_speech_end_frame:
            chosen = spans[group_index]
            out_path = (
                output_dir
                / chosen.folder
                / f"frame_{frame_index:09d}.{ext}"
            )

            if ext in {"jpg", "jpeg"}:
                written = cv2.imwrite(
                    str(out_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
                )
            else:
                written = cv2.imwrite(str(out_path), frame)

            if not written:
                cap.release()
                raise RuntimeError(f"Failed to write frame: {out_path}")

            assigned_frames += 1

        frame_index += 1

    cap.release()
    frames_decoded = frame_index

    groups = _build_inclusive_groups(spans, fps, frames_decoded)

    # Derive token frame metadata from the exact same ranges exposed in groups.
    for span, (group_start, group_end) in zip(spans, groups):
        if group_end >= group_start:
            span.first_frame = group_start
            span.last_frame = group_end
            span.frame_count = group_end - group_start + 1
        else:
            span.first_frame = None
            span.last_frame = None
            span.frame_count = 0

    manifest = {
        "video": str(video_path),
        "text": transcript,
        "fps": fps,
        "groups": groups,
        "groups_format": (
            "Inclusive source-frame ranges. groups[i] corresponds to tokens[i]. "
            "Adjacent ranges satisfy groups[i+1][0] == groups[i][1] + 1. "
            "An empty range [s, s-1] means no unique source frame was available "
            "for that token."
        ),
        "frame_assignment_policy": (
            "A source frame is written at most once. If adjacent spoken-token "
            "windows overlap a source frame, the later token gets that frame. "
            "Silence between tokens is preferentially attached to the next "
            "token. Group 0 begins at frame 0; video after the final spoken "
            "token is omitted."
        ),
        "total_frames_reported": total_frames_reported,
        "frames_decoded": frames_decoded,
        "frames_assigned_to_tokens": assigned_frames,
        "width": width,
        "height": height,
        "tokens": [
            {
                "index": span.index,
                "token_id": span.token_id,
                "token": span.token,
                "text": span.text,
                "char_start": span.char_start,
                "char_end": span.char_end,
                "time_start": span.time_start,
                "time_end": span.time_end,
                "frame_count": span.frame_count,
                "first_frame": span.first_frame,
                "last_frame": span.last_frame,
                "folder": span.folder,
            }
            for span in spans
        ],
    }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def group_frames_by_token(
    video_path: str | Path,
    tokenizer: Any,
    output_dir: str | Path,
    *,
    whisper_model: str = "small",
    language: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "default",
    image_ext: str = "jpg",
    jpeg_quality: int = 95,
    whisper_instance: Optional[WhisperModel] = None,
) -> dict[str, Any]:
    """
    Transcribe, align and save one video's token frames and metadata.

    Example:
        from transformers import AutoTokenizer
        from grouper import group_frames_by_token

        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            use_fast=True,
        )

        manifest = group_frames_by_token(
            "input.mp4",
            tokenizer,
            "token_frames",
        )
    """
    transcript, words = transcribe_words(
        video_path,
        whisper_model=whisper_model,
        language=language,
        device=device,
        compute_type=compute_type,
        whisper_instance=whisper_instance,
    )

    if not words:
        raise RuntimeError("No timestamped speech was found in the video.")

    spans = align_tokens(transcript, words, tokenizer)
    if not spans:
        raise RuntimeError("Tokenizer produced no alignable tokens.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "transcript.txt").open("w", encoding="utf-8") as f:
        f.write(transcript)

    with (output_dir / "words.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "text": word.text,
                    "char_start": word.char_start,
                    "char_end": word.char_end,
                    "time_start": word.time_start,
                    "time_end": word.time_end,
                }
                for word in words
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    return save_grouped_frames(
        video_path,
        spans,
        output_dir,
        transcript=transcript,
        image_ext=image_ext,
        jpeg_quality=jpeg_quality,
    )


def _jsonl_record(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build a dataset row from a video's transcript and token frame groups."""
    return {
        "video": manifest["video"],
        "text": manifest["text"],
        "windows": manifest["groups"],
    }


def write_jsonl(
    records: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write one compact JSON object per successfully processed video."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    return output_path


def find_videos(
    input_path: str | Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    """Return a file input or all recognized videos in a directory."""
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    if input_path.is_file():
        # Accept any file extension here; the decoder determines readability.
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Input is neither a file nor a directory: {input_path}")

    candidates = input_path.rglob("*") if recursive else input_path.iterdir()
    videos = sorted(
        (
            path
            for path in candidates
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: str(path.relative_to(input_path)).casefold(),
    )

    if not videos:
        scope = "or its subdirectories" if recursive else "the directory"
        extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise ValueError(
            f"No recognized video files found in {scope}: {input_path}. "
            f"Supported extensions: {extensions}"
        )

    return videos


def group_folder_by_token(
    input_dir: str | Path,
    tokenizer: Any,
    output_dir: str | Path,
    *,
    recursive: bool = False,
    whisper_model: str = "small",
    language: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "default",
    image_ext: str = "jpg",
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    """
    Process every recognized video in a directory.

    Each video's output is placed below ``output_dir`` using its path relative
    to ``input_dir``. The filename (including its extension) is the final
    directory name, so videos such as ``clip.mp4`` and ``clip.mkv`` cannot
    overwrite each other.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Folder input must be a directory: {input_dir}")

    videos = find_videos(input_dir, recursive=recursive)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loading Whisper is relatively expensive, so share one model across the
    # entire batch instead of recreating it for every video.
    whisper_instance = WhisperModel(
        whisper_model,
        device=device,
        compute_type=compute_type,
    )

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    jsonl_records: list[dict[str, Any]] = []

    for position, video_path in enumerate(videos, start=1):
        relative_video = video_path.relative_to(input_dir)
        video_output = output_dir / relative_video
        print(f"[{position}/{len(videos)}] Processing: {video_path}")

        try:
            manifest = group_frames_by_token(
                video_path,
                tokenizer,
                video_output,
                whisper_model=whisper_model,
                language=language,
                device=device,
                compute_type=compute_type,
                image_ext=image_ext,
                jpeg_quality=jpeg_quality,
                whisper_instance=whisper_instance,
            )
        except Exception as exc:
            failures.append({"video": str(video_path), "error": str(exc)})
            print(f"  Failed: {exc}", file=sys.stderr)
            continue

        results.append(
            {
                "video": str(video_path),
                "output": str(video_output),
                "manifest": str(video_output / "manifest.json"),
                "token_occurrences": len(manifest["tokens"]),
                "frames_assigned_to_tokens": manifest[
                    "frames_assigned_to_tokens"
                ],
            }
        )
        jsonl_records.append(_jsonl_record(manifest))
        print(
            f"  Done: {len(manifest['tokens'])} token occurrences, "
            f"{manifest['frames_assigned_to_tokens']} assigned frames."
        )

    jsonl_path = write_jsonl(
        jsonl_records,
        output_dir / JSONL_FILENAME,
    )

    batch_manifest = {
        "input_directory": str(input_dir),
        "recursive": recursive,
        "videos_found": len(videos),
        "videos_succeeded": len(results),
        "videos_failed": len(failures),
        "jsonl": str(jsonl_path),
        "results": results,
        "failures": failures,
    }

    with (output_dir / "batch_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(batch_manifest, f, ensure_ascii=False, indent=2)

    return batch_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Group video frames by the tokenizer token being spoken. The "
            "input can be one video or a directory containing videos. "
            "Frames are never duplicated between token groups; if a frame "
            "overlaps two token windows, the later token wins."
        )
    )

    parser.add_argument(
        "input",
        help="Input video or directory containing videos",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help=(
            "Hugging Face fast tokenizer name or local path, "
            "e.g. bert-base-uncased"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="token_frames",
        help="Output directory (default: token_frames)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also find videos in subdirectories of a folder input",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        help="faster-whisper model name/size (default: small)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code such as en or es; default is auto-detect",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Whisper device (default: auto)",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help=(
            "CTranslate2 compute type, e.g. default, float16, int8, "
            "int8_float16"
        ),
    )
    parser.add_argument(
        "--image-ext",
        default="jpg",
        choices=["jpg", "jpeg", "png"],
        help="Saved frame format (default: jpg)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality 0-100 (default: 95)",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not 0 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be between 0 and 100")

    input_path = Path(args.input)
    try:
        find_videos(input_path, recursive=args.recursive)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    if input_path.is_dir():
        batch_manifest = group_folder_by_token(
            input_path,
            tokenizer,
            args.output,
            recursive=args.recursive,
            whisper_model=args.whisper_model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            image_ext=args.image_ext,
            jpeg_quality=args.jpeg_quality,
        )
        print(
            f"Batch complete: {batch_manifest['videos_succeeded']} succeeded, "
            f"{batch_manifest['videos_failed']} failed."
        )
        print(f"Output: {Path(args.output).resolve()}")
        print(
            "Batch manifest: "
            f"{(Path(args.output) / 'batch_manifest.json').resolve()}"
        )
        print(f"JSONL: {(Path(args.output) / JSONL_FILENAME).resolve()}")
        if batch_manifest["videos_failed"]:
            raise SystemExit(1)
    else:
        manifest = group_frames_by_token(
            input_path,
            tokenizer,
            args.output,
            whisper_model=args.whisper_model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            image_ext=args.image_ext,
            jpeg_quality=args.jpeg_quality,
        )
        jsonl_path = write_jsonl(
            [_jsonl_record(manifest)],
            Path(args.output) / JSONL_FILENAME,
        )

        print(
            f"Done: {len(manifest['tokens'])} token occurrences, "
            f"{manifest['frames_assigned_to_tokens']} assigned frames."
        )
        print(f"Output: {Path(args.output).resolve()}")
        print(f"Manifest: {(Path(args.output) / 'manifest.json').resolve()}")
        print(f"JSONL: {jsonl_path.resolve()}")


if __name__ == "__main__":
    main()
