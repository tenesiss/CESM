"""Tiled, pre-dropout attention statistics with a recomputing backward pass.

This is a portable PyTorch implementation of online softmax, not a fused CUDA
kernel. Scores and probabilities are computed in query/frame tiles and discarded
between tiles. Q/K, mask/window metadata and O(B*H*T) row statistics are saved for
backward. Only first-order gradients are supported.
"""

from typing import Optional, Tuple

import torch
from torch.autograd.function import once_differentiable


class _AttentionStatistics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, key_counts, windows, scale, query_block_size, frame_block_size):
        B, H, T, _ = q.shape
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        shape = (B, H, T)
        row_max = torch.empty(shape, device=q.device, dtype=dtype)
        inv_sum = torch.empty_like(row_max)
        mean_frame = torch.empty_like(row_max)
        window_mass = torch.empty_like(row_max) if windows is not None else None

        # Explicitly disable ambient AMP: dot products, softmax accumulators and
        # backward reductions must stay in FP32 (FP64 for numerical checks).
        with torch.autocast(device_type=q.device.type, enabled=False):
            for t0 in range(0, T, query_block_size):
                t1 = min(T, t0 + query_block_size)
                qt = q[:, :, t0:t1].to(dtype)
                m = torch.full((B, H, t1 - t0), -torch.inf, device=q.device, dtype=dtype)
                z = torch.zeros_like(m)
                u = torch.zeros_like(m)
                w = torch.zeros_like(m) if windows is not None else None
                counts = key_counts[:, None, t0:t1, None]
                if windows is not None:
                    starts = windows[:, None, t0:t1, 0, None]
                    ends = windows[:, None, t0:t1, 1, None]
                for f0 in range(0, k.shape[2], frame_block_size):
                    f1 = min(k.shape[2], f0 + frame_block_size)
                    frames = torch.arange(f0, f1, device=q.device)
                    scores = (qt @ k[:, :, f0:f1].to(dtype).transpose(-2, -1)) * scale
                    scores.masked_fill_(frames >= counts, -torch.inf)
                    next_m = torch.maximum(m, scores.amax(dim=-1))
                    # All-masked rows have zero weights, including when every
                    # frame block is masked. Avoid -inf - -inf in the rescale.
                    safe_m = next_m.masked_fill(torch.isneginf(next_m), 0.0)
                    rescale = (m - safe_m).exp()
                    weights = (scores - safe_m.unsqueeze(-1)).exp()
                    z = rescale * z + weights.sum(dim=-1)
                    u = rescale * u + (weights * frames.to(dtype)).sum(dim=-1)
                    if windows is not None:
                        inside = (frames >= starts) & (frames <= ends)
                        w = rescale * w + (weights * inside).sum(dim=-1)
                    m = next_m
                inv_z = z.clamp_min(1.0).reciprocal()
                row_max[:, :, t0:t1] = m.masked_fill(torch.isneginf(m), 0.0)
                inv_sum[:, :, t0:t1] = inv_z
                mean_frame[:, :, t0:t1] = u * inv_z
                if windows is not None:
                    window_mass[:, :, t0:t1] = w * inv_z

        ctx.save_for_backward(q, k, key_counts, windows, row_max, inv_sum, mean_frame, window_mass)
        ctx.scale = scale
        ctx.query_block_size = query_block_size
        ctx.frame_block_size = frame_block_size
        ctx.set_materialize_grads(False)
        # Normalize each head before averaging; raw numerators/denominators
        # cannot be averaged because each head has its own softmax normalizer.
        return mean_frame.mean(dim=1), (
            window_mass.mean(dim=1) if window_mass is not None else mean_frame.new_empty(0)
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_frame, grad_mass):
        q, k, key_counts, windows, row_max, inv_sum, mean_frame, window_mass = ctx.saved_tensors
        dtype = mean_frame.dtype
        H, T = q.shape[1:3]
        dq = torch.zeros_like(q, dtype=dtype) if ctx.needs_input_grad[0] else None
        dk = torch.zeros_like(k, dtype=dtype) if ctx.needs_input_grad[1] else None
        if grad_frame is None and grad_mass is None:
            return dq, dk, None, None, None, None, None

        with torch.autocast(device_type=q.device.type, enabled=False):
            for t0 in range(0, T, ctx.query_block_size):
                t1 = min(T, t0 + ctx.query_block_size)
                qt = q[:, :, t0:t1].to(dtype)
                counts = key_counts[:, None, t0:t1, None]
                g_frame = grad_frame[:, None, t0:t1, None].to(dtype) / H if grad_frame is not None else None
                g_mass = grad_mass[:, None, t0:t1, None].to(dtype) / H if grad_mass is not None else None
                if windows is not None and g_mass is not None:
                    starts = windows[:, None, t0:t1, 0, None]
                    ends = windows[:, None, t0:t1, 1, None]
                for f0 in range(0, k.shape[2], ctx.frame_block_size):
                    f1 = min(k.shape[2], f0 + ctx.frame_block_size)
                    kt = k[:, :, f0:f1].to(dtype)
                    frames = torch.arange(f0, f1, device=q.device)
                    scores = (qt @ kt.transpose(-2, -1)) * ctx.scale
                    scores.masked_fill_(frames >= counts, -torch.inf)
                    # Keep max and inverse sum separate to avoid losing the
                    # log-normalizer correction when logits are very large.
                    probs = (scores - row_max[:, :, t0:t1, None]).exp()
                    probs *= inv_sum[:, :, t0:t1, None]
                    ds = torch.zeros_like(probs)
                    if g_frame is not None:
                        ds += g_frame * (frames.to(dtype) - mean_frame[:, :, t0:t1, None])
                    if windows is not None and g_mass is not None:
                        inside = (frames >= starts) & (frames <= ends)
                        ds += g_mass * (inside.to(dtype) - window_mass[:, :, t0:t1, None])
                    ds *= probs
                    ds *= ctx.scale
                    if dq is not None:
                        dq[:, :, t0:t1] += ds @ kt
                    if dk is not None:
                        dk[:, :, f0:f1] += ds.transpose(-2, -1) @ qt

        return (
            dq.to(q.dtype) if dq is not None else None,
            dk.to(k.dtype) if dk is not None else None,
            None, None, None, None, None,
        )


def attention_statistics_from_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    token_video_end: torch.Tensor,
    text_lengths: torch.Tensor,
    video_lengths: torch.Tensor,
    windows: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = None,
    query_block_size: int = 32,
    frame_block_size: int = 128,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return head-averaged expected frames and optional window masses [B,T].

    Q/K have shapes [B,H,T,D] and [B,H,F,D]. Visibility is the same prefix mask
    as CESM token attention: query t sees frames <= token_video_end[t], clipped
    to video_lengths; padded text queries see nothing. Windows are inclusive.
    No attention dropout is applied. FP16/BF16 inputs accumulate in FP32.
    """
    if query_block_size <= 0 or frame_block_size <= 0:
        raise ValueError("attention block sizes must be positive")
    key_counts = torch.minimum(token_video_end + 1, video_lengths[:, None]).clamp(0, k.shape[2])
    valid = torch.arange(q.shape[2], device=q.device)[None, :] < text_lengths[:, None]
    key_counts = key_counts.masked_fill(~valid, 0)
    if windows is not None:
        last_frame = (video_lengths - 1).clamp_min(0)[:, None]
        starts = torch.minimum(windows[..., 0].clamp_min(0), last_frame)
        ends = torch.maximum(starts, torch.minimum(windows[..., 1], last_frame))
        windows = torch.stack((starts, ends), dim=-1)
    mean_frame, mass = _AttentionStatistics.apply(
        q, k, key_counts, windows, q.shape[-1] ** -0.5 if scale is None else scale,
        query_block_size, frame_block_size,
    )
    return mean_frame, mass if windows is not None else None
