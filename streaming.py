"""Shared causal streaming decoder and commit options for inference and evaluation."""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Optional

import torch


def add_decode_arguments(parser, *, prefix=""):
    """Expose identical decoder defaults, optionally under --accuracy-* flags."""
    def add(*flags, **kwargs):
        flags = tuple("--" + prefix + flag[2:] for flag in flags)
        if "dest" in kwargs:
            kwargs["dest"] = prefix.replace("-", "_") + kwargs["dest"]
        if kwargs.get("help") not in (None, argparse.SUPPRESS):
            kwargs["help"] = kwargs["help"].replace("--", "--" + prefix)
        parser.add_argument(*flags, **kwargs)

    # Learned confidence is the primary token-commit decision. With
    # --confidence-relax-per-frame 0 this becomes a simple hard threshold.
    add("--confidence-threshold", type=float, default=0.75,
                   help="Base confidence required to commit a token")
    add("--confidence-min-threshold", type=float, default=0.45,
                   help="Floor for the relaxed confidence threshold")
    add("--confidence-relax-per-frame", type=float, default=0.01,
                   help="Subtract this from the threshold for each frame after --confidence-relax-after")
    add("--confidence-relax-after", type=int, default=4,
                   help="Frames since the previous commit before threshold relaxation starts")
    add("--stable-frames", type=int, default=1,
                   help="Optional argmax stability guard; 1 lets confidence alone decide timing")
    add("--warmup-frames", type=int, default=4)
    add("--min-frames-per-token", type=int, default=2)
    add("--repeat-token-cooldown-frames", type=int, default=10,
                   help="Minimum frame-index gap since the same token ID was last emitted; "
                        "skip earlier predictions even if other tokens intervened (default: 5; 0 disables)")
    add("--min-token-prob", type=float, default=0.0,
                   help="Optional sanity floor for next-token probability; 0 disables it")
    # Backward-compatible spelling from the first prototype.
    add("--emit-prob", dest="min_token_prob", type=float, help=argparse.SUPPRESS)
    add("--token-temperature", type=float, default=None,
                   help="Override tau_token; default uses the confidence-training value")
    add("--conf-temperature", type=float, default=None,
                   help="Override tau_conf; default uses the confidence-training value")
    add("--max-tokens", type=int, default=512)
    add("--flush-tokens", type=int, default=0)


def validate_decode_arguments(args):
    for name in ("confidence_threshold", "confidence_min_threshold", "min_token_prob"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    if args.confidence_min_threshold > args.confidence_threshold:
        raise ValueError("--confidence-min-threshold must be in [0, confidence-threshold]")
    if not math.isfinite(args.confidence_relax_per_frame) or args.confidence_relax_per_frame < 0:
        raise ValueError("--confidence-relax-per-frame must be finite and >= 0")
    for name in ("confidence_relax_after", "min_frames_per_token", "repeat_token_cooldown_frames",
                 "warmup_frames", "flush_tokens", "stable_frames", "max_tokens"):
        minimum = 1 if name in ("stable_frames", "max_tokens") else 0
        if getattr(args, name) < minimum:
            raise ValueError(f"--{name.replace('_', '-')} must be >= {minimum}")
    for name in ("token_temperature", "conf_temperature"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and > 0")


def mask_invalid_generation_logits(logits: torch.Tensor, tokenizer) -> torch.Tensor:
    logits = logits.clone()
    # PAD is never valid. BOS is masked when it is a dedicated token; several
    # causal LMs intentionally share BOS and EOS, in which case masking it would
    # make end-of-sequence generation impossible. UNK remains legal.
    logits[..., tokenizer.pad_id] = -torch.inf
    if tokenizer.bos_id != tokenizer.eos_id:
        logits[..., tokenizer.bos_id] = -torch.inf
    return logits


def decode_stream(model, tokenizer, frames, device, args, *, token_temperature=1.0,
                  conf_temperature=1.0, amp=False, fps=25.0, on_event=None,
                  debug_every=0, pace_realtime=False):
    """Decode preprocessed frames lazily, starting with BOS and feeding back commits only.

    Returns (emitted token IDs excluding EOS, processed frame count). The caller
    owns model eval mode and the frame iterator's resources.
    """
    amp_enabled = bool(amp and device.type == "cuda")

    def emit(event):
        if on_event is not None:
            on_event(event)

    state = model.init_stream_state()

    # Encode BOS once, then update text caches only after commits. Token-query
    # variants reuse its query as video K/V grows; frame fusion queries the
    # cached text from each arriving frame.
    bos = torch.tensor([tokenizer.bos_id], dtype=torch.long, device=device)
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
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
    last_emit_frame_by_token: dict[int, int] = {}
    ended = False

    frame_iterator = iter(frames)
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        while True:
            frame_wall = time.perf_counter()
            try:
                frame = next(frame_iterator)
            except StopIteration:
                break
            frame_idx += 1

            frame = frame.unsqueeze(0).to(device, non_blocking=True)
            conf_logit, state = model.stream_push_video_frame(frame, state)
            token_logits, attn = model.stream_predict_next(state)
            token_logits = mask_invalid_generation_logits(token_logits[:, -1], tokenizer)
            probs = torch.softmax(token_logits.float() / token_temperature, dim=-1)
            prob, tok = probs.max(dim=-1)
            tok_id = int(tok.item())
            tok_prob = float(prob.item())
            conf_prob = float(torch.sigmoid(conf_logit[:, -1].float() / conf_temperature).item())

            frames_since_commit = frame_idx - last_emit_frame
            enough_gap = frames_since_commit >= args.min_frames_per_token
            warm = frame_idx + 1 >= args.warmup_frames
            probability_ok = tok_prob >= args.min_token_prob
            token_last_emit_frame = last_emit_frame_by_token.get(tok_id)
            repeat_ok = (
                token_last_emit_frame is None
                or frame_idx - token_last_emit_frame >= args.repeat_token_cooldown_frames
            )

            # A transient confidence drop must not reset a stable candidate's
            # peak. Changing argmax or failing a guard does reset it. A token
            # in cooldown skips this frame's prediction without choosing a
            # runner-up or accumulating confidence/stability for a later commit.
            # Video caches, diagnostics and real-time pacing still advance.
            if probability_ok and repeat_ok:
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
                    emit({
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
                last_emit_frame_by_token[committed] = frame_idx
                token_text = tokenizer.decode([committed])
                emit({
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

            if debug_every > 0 and frame_idx % debug_every == 0:
                top_text = tokenizer.decode([tok_id]) if tok_id != tokenizer.eos_id else "<eos>"
                # Expected video-frame index under text->video attention.
                A = attn.mean(dim=1)[0, 0]  # [cached_frames]
                positions = torch.arange(A.numel(), device=A.device, dtype=A.dtype)
                mu = float((A * positions).sum().item())
                print(
                    f"\n[frame={frame_idx} t={frame_idx/fps:.2f}s next={top_text!r} "
                    f"p={tok_prob:.3f} conf={conf_prob:.3f} peak_conf={candidate_peak_conf:.3f} "
                    f"commit_thr={effective_conf_threshold:.3f} since_commit={frames_since_commit} "
                    f"repeat_ok={repeat_ok} "
                    f"attn_mu={mu:.1f}]",
                    file=sys.stderr,
                    flush=True,
                )

            if pace_realtime:
                target = 1.0 / fps
                elapsed = time.perf_counter() - frame_wall
                if elapsed < target:
                    time.sleep(target - elapsed)

    # Optional greedy flush bypasses confidence using the final video K/V cache.
    # Frame fusion is excluded: new token logits require a new video frame.
    if (model.cfg.text_fusion != "frame" and not ended
            and args.flush_tokens > 0 and len(emitted_ids) < args.max_tokens):
        with torch.inference_mode(), torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            for _ in range(args.flush_tokens):
                logits, _ = model.stream_predict_next(state)
                logits = mask_invalid_generation_logits(logits[:, -1], tokenizer)
                probs = torch.softmax(logits.float() / token_temperature, dim=-1)
                prob, tok = probs.max(dim=-1)
                tid = int(tok.item())
                if tid == tokenizer.eos_id:
                    break
                token_last_emit_frame = last_emit_frame_by_token.get(tid)
                if (token_last_emit_frame is not None
                        and frame_idx - token_last_emit_frame < args.repeat_token_cooldown_frames):
                    # Flushing cannot advance the frame clock or clear a cooldown.
                    break
                emitted_ids.append(tid)
                last_emit_frame_by_token[tid] = frame_idx
                emit({
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

    return emitted_ids, frame_idx + 1
