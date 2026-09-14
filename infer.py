#!/usr/bin/env python3
"""
Incremental / streaming-style inference for the Unnoba prototype.

The input is a normal video file, but frames are decoded and processed one at a
time. No future frame is read by the neural network before producing the current
next-token distribution.

Caches used during inference:
  * causal Conv3D raw-frame history
  * video Conformer self-attention K/V per layer
  * video causal-convolution history per layer
  * text encoder self-attention K/V per layer
  * Mixed Block text->video cross-attention K/V (grows per frame)
  * Mixed Block video->text cross-attention K/V (grows per committed token)
  * Mixed Block video-path self-attention K/V

Token commitment is driven by the learned per-frame confidence head. The base
confidence threshold can optionally relax as more frames pass since the previous
commit, which prevents a low-confidence token from stalling the stream forever.
While the same next-token candidate remains active, its peak confidence is kept;
this lets a later relaxed threshold accept a confidence peak that occurred near
the true boundary. A minimum token probability and candidate-stability count are
available only as optional safety guards; confidence is the primary commit gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# train.py intentionally doubles as the model module so the requested prototype
# remains exactly two scripts.
from train import FixedMouthCropper, ModelConfig, UnnobaModel, tokenizer_from_state_dict


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


def mask_invalid_generation_logits(logits: torch.Tensor, tokenizer) -> torch.Tensor:
    logits = logits.clone()
    # PAD is never valid. BOS is masked when it is a dedicated token; several
    # causal LMs intentionally share BOS and EOS, in which case masking it would
    # make end-of-sequence generation impossible. UNK remains legal.
    logits[..., tokenizer.pad_id] = -torch.inf
    if tokenizer.bos_id != tokenizer.eos_id:
        logits[..., tokenizer.bos_id] = -torch.inf
    return logits


def emit_event(args, event: dict) -> None:
    if args.json_events:
        print(json.dumps(event, ensure_ascii=False), flush=True)
    else:
        token = event.get("token", "")
        if token:
            sys.stdout.write(token)
            sys.stdout.flush()


def main(args) -> None:
    if not (0.0 <= args.confidence_threshold <= 1.0):
        raise SystemExit("--confidence-threshold must be in [0,1]")
    if not (0.0 <= args.confidence_min_threshold <= args.confidence_threshold):
        raise SystemExit("--confidence-min-threshold must be in [0, confidence-threshold]")
    if args.confidence_relax_per_frame < 0:
        raise SystemExit("--confidence-relax-per-frame must be >= 0")
    if args.confidence_relax_after < 0 or args.min_frames_per_token < 0:
        raise SystemExit("frame-count commit arguments must be >= 0")
    if args.stable_frames < 1:
        raise SystemExit("--stable-frames must be >= 1")
    if not (0.0 <= args.min_token_prob <= 1.0):
        raise SystemExit("--min-token-prob must be in [0,1]")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, tokenizer, preprocess, ckpt = load_model(args.checkpoint, device)
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
    state = model.init_stream_state()

    # The first autoregressive query is BOS. It is encoded exactly once; while
    # frames arrive we only rerun its Mixed-Block cross-attention query against
    # the growing cached video K/V. The text self-attention cache is not touched
    # again until a token is actually committed.
    bos = torch.tensor([tokenizer.bos_id], dtype=torch.long, device=device)
    with torch.inference_mode():
        state = model.stream_push_text_token(bos, state)

    candidate: Optional[int] = None
    candidate_streak = 0
    candidate_peak_conf = 0.0
    candidate_peak_prob = 0.0
    candidate_peak_frame = -1
    emitted_ids = []
    frame_idx = -1
    # Treat stream start like a virtual commit immediately before frame 0.
    last_emit_frame = -1
    ended = False
    started_wall = time.perf_counter()

    if args.json_events:
        print(json.dumps({
            "event": "start",
            "video": str(Path(args.video)),
            "fps": fps,
            "device": str(device),
            "checkpoint_epoch": ckpt.get("epoch"),
            "confidence_epoch": ckpt.get("confidence_epoch"),
            "confidence_trained": bool(confidence_meta.get("trained", False)),
            "confidence_threshold": args.confidence_threshold,
            "confidence_min_threshold": args.confidence_min_threshold,
            "confidence_relax_per_frame": args.confidence_relax_per_frame,
            "token_temperature": token_temperature,
            "confidence_temperature": conf_temperature,
        }), flush=True)

    with torch.inference_mode():
        while True:
            frame_wall = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1

            frame = cropper(frame_bgr).unsqueeze(0).to(device, non_blocking=True)
            conf_logit, state = model.stream_push_video_frame(frame, state)
            token_logits, attn = model.stream_predict_next(state)
            token_logits = mask_invalid_generation_logits(token_logits[:, -1], tokenizer)
            probs = torch.softmax(token_logits / token_temperature, dim=-1)
            prob, tok = probs.max(dim=-1)
            tok_id = int(tok.item())
            tok_prob = float(prob.item())
            conf_prob = float(torch.sigmoid(conf_logit[:, -1] / conf_temperature).item())

            frames_since_commit = frame_idx - last_emit_frame
            enough_gap = frames_since_commit >= args.min_frames_per_token
            warm = frame_idx + 1 >= args.warmup_frames
            probability_ok = tok_prob >= args.min_token_prob

            # Track stability independently of confidence. Confidence may peak
            # for one frame at the learned boundary, so dropping the candidate
            # merely because the next frame is less confident would throw away
            # exactly the signal we trained the head to produce.
            if probability_ok:
                if candidate == tok_id:
                    candidate_streak += 1
                    if conf_prob > candidate_peak_conf:
                        candidate_peak_conf = conf_prob
                        candidate_peak_prob = tok_prob
                        candidate_peak_frame = frame_idx
                else:
                    candidate = tok_id
                    candidate_streak = 1
                    candidate_peak_conf = conf_prob
                    candidate_peak_prob = tok_prob
                    candidate_peak_frame = frame_idx
            else:
                candidate = None
                candidate_streak = 0
                candidate_peak_conf = 0.0
                candidate_peak_prob = 0.0
                candidate_peak_frame = -1

            relax_frames = max(0, frames_since_commit - args.confidence_relax_after)
            effective_conf_threshold = max(
                args.confidence_min_threshold,
                args.confidence_threshold - args.confidence_relax_per_frame * relax_frames,
            )
            confidence_ok = candidate is not None and candidate_peak_conf >= effective_conf_threshold

            if (
                warm
                and enough_gap
                and confidence_ok
                and candidate_streak >= args.stable_frames
            ):
                committed = candidate
                committed_prob = tok_prob
                committed_conf = candidate_peak_conf
                committed_conf_frame = candidate_peak_frame
                committed_peak_prob = candidate_peak_prob
                committed_threshold = effective_conf_threshold
                candidate = None
                candidate_streak = 0
                candidate_peak_conf = 0.0
                candidate_peak_prob = 0.0
                candidate_peak_frame = -1
                last_emit_frame = frame_idx

                if committed == tokenizer.eos_id:
                    ended = True
                    if args.json_events:
                        emit_event(args, {
                            "event": "eos",
                            "frame": frame_idx,
                            "time_s": frame_idx / fps,
                            "prob": committed_prob,
                            "confidence": committed_conf,
                            "confidence_frame": committed_conf_frame,
                            "confidence_threshold": committed_threshold,
                        })
                    break

                emitted_ids.append(committed)
                token_text = tokenizer.decode([committed])
                emit_event(args, {
                    "event": "token",
                    "frame": frame_idx,
                    "time_s": frame_idx / fps,
                    "token_id": committed,
                    "token": token_text,
                    "prob": committed_prob,
                    "confidence": committed_conf,
                    "confidence_frame": committed_conf_frame,
                    "confidence_peak_token_prob": committed_peak_prob,
                    "confidence_threshold": committed_threshold,
                    "frames_since_previous_commit": frames_since_commit,
                })

                # Advance the causal text encoder only after commitment. This
                # appends one K/V entry per text layer and one projected text K/V
                # entry for the Mixed Block's video->text cross-attention cache.
                tid = torch.tensor([committed], dtype=torch.long, device=device)
                state = model.stream_push_text_token(tid, state)

                if len(emitted_ids) >= args.max_tokens:
                    break

            if args.debug_every > 0 and frame_idx % args.debug_every == 0:
                top_text = tokenizer.decode([tok_id]) if tok_id != tokenizer.eos_id else "<eos>"
                # Mean attention position of the current next-token query.
                A = attn.mean(dim=1)[0, 0]  # [cached_frames]
                positions = torch.arange(A.numel(), device=A.device, dtype=A.dtype)
                mu = float((A * positions).sum().item())
                print(
                    f"\n[frame={frame_idx} t={frame_idx/fps:.2f}s next={top_text!r} "
                    f"p={tok_prob:.3f} conf={conf_prob:.3f} peak_conf={candidate_peak_conf:.3f} "
                    f"commit_thr={effective_conf_threshold:.3f} since_commit={frames_since_commit} "
                    f"attn_mu={mu:.1f}]",
                    file=sys.stderr,
                    flush=True,
                )

            if args.pace_realtime:
                target = 1.0 / fps
                elapsed = time.perf_counter() - frame_wall
                if elapsed < target:
                    time.sleep(target - elapsed)

    cap.release()

    # Optional end-of-video greedy flush. This reuses the final video K/V cache;
    # no video is recomputed. It is disabled by default because it is not truly
    # streaming behavior, but it is useful when evaluating an under-confident
    # early prototype.
    if (model.cfg.text_fusion != "frame" and not ended
            and args.flush_tokens > 0 and len(emitted_ids) < args.max_tokens):
        with torch.inference_mode():
            for _ in range(args.flush_tokens):
                logits, _ = model.stream_predict_next(state)
                logits = mask_invalid_generation_logits(logits[:, -1], tokenizer)
                probs = torch.softmax(logits / token_temperature, dim=-1)
                prob, tok = probs.max(dim=-1)
                tid = int(tok.item())
                if tid == tokenizer.eos_id:
                    break
                emitted_ids.append(tid)
                emit_event(args, {
                    "event": "flush_token",
                    "frame": frame_idx,
                    "time_s": frame_idx / fps if frame_idx >= 0 else 0.0,
                    "token_id": tid,
                    "token": tokenizer.decode([tid]),
                    "prob": float(prob.item()),
                })
                state = model.stream_push_text_token(
                    torch.tensor([tid], dtype=torch.long, device=device), state
                )
                if len(emitted_ids) >= args.max_tokens:
                    break

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
    p.add_argument("--no-face-detector", action="store_true")

    # Learned confidence is the primary token-commit decision. With
    # --confidence-relax-per-frame 0 this becomes a simple hard threshold.
    p.add_argument("--confidence-threshold", type=float, default=0.75,
                   help="Base confidence required to commit a token")
    p.add_argument("--confidence-min-threshold", type=float, default=0.45,
                   help="Floor for the relaxed confidence threshold")
    p.add_argument("--confidence-relax-per-frame", type=float, default=0.01,
                   help="Subtract this from the threshold for each frame after --confidence-relax-after")
    p.add_argument("--confidence-relax-after", type=int, default=4,
                   help="Frames since the previous commit before threshold relaxation starts")
    p.add_argument("--stable-frames", type=int, default=1,
                   help="Optional argmax stability guard; 1 lets confidence alone decide timing")
    p.add_argument("--warmup-frames", type=int, default=3)
    p.add_argument("--min-frames-per-token", type=int, default=2)
    p.add_argument("--min-token-prob", type=float, default=0.0,
                   help="Optional sanity floor for next-token probability; 0 disables it")
    # Backward-compatible spelling from the first prototype.
    p.add_argument("--emit-prob", dest="min_token_prob", type=float, help=argparse.SUPPRESS)
    p.add_argument("--token-temperature", type=float, default=None,
                   help="Override tau_token; default uses the checkpoint confidence-training value")
    p.add_argument("--conf-temperature", type=float, default=None,
                   help="Override tau_conf; default uses the checkpoint confidence-training value")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--flush-tokens", type=int, default=0)

    p.add_argument("--pace-realtime", action="store_true",
                   help="Sleep when processing is faster than the source FPS; otherwise run as fast as possible")
    p.add_argument("--json-events", action="store_true",
                   help="Emit one JSON object per token instead of a live plain-text stream")
    p.add_argument("--debug-every", type=int, default=0,
                   help="Print current next-token/attention diagnostics to stderr every N frames")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
