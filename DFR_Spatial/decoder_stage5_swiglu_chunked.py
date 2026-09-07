"""Official CHUNKED_EAGER Stage-5 SwiGLU residual.

This is the small runtime subset of LTX-core's transformer/swiglu.py and
transformer/chunked/mlp.py required by the Spatial DFR decoder.  It keeps one
16,384-token workspace and uses the official fused BF16 Triton gate/up kernel
when eligible, with the source PyTorch path as its fallback.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_SWIGLU_TILE_SIZE = 16_384

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the Comfy runtime
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


def triton_swiglu_available() -> bool:
    return bool(_TRITON_AVAILABLE and torch.cuda.is_available())


def _cuda_tensors_for_triton(*tensors: torch.Tensor) -> bool:
    if not tensors or not tensors[0].is_cuda:
        return False
    device = tensors[0].device
    return all(t.is_cuda and t.device == device for t in tensors)


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_up_mul_kernel(
        x_ptr,
        w_ptr,
        g_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_gm,
        stride_gn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            w = tl.load(
                w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += tl.dot(x, w)
        g = tl.load(
            g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        tl.store(
            g_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn,
            g * acc.to(g.dtype),
            mask=mask_m[:, None] & mask_n[None, :],
        )


    @triton.jit
    def _fused_gate_up_swiglu_kernel(
        x_ptr,
        w_gate_ptr,
        w_up_ptr,
        out_ptr,
        M,
        K,
        N,
        stride_xm,
        stride_xk,
        stride_gate_n,
        stride_gate_k,
        stride_up_n,
        stride_up_k,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            gate_w = tl.load(
                w_gate_ptr + offs_n[None, :] * stride_gate_n + offs_k[:, None] * stride_gate_k,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            up_w = tl.load(
                w_up_ptr + offs_n[None, :] * stride_up_n + offs_k[:, None] * stride_up_k,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            gate_acc += tl.dot(x, gate_w)
            up_acc += tl.dot(x, up_w)
        silu_bf16 = (gate_acc * tl.sigmoid(gate_acc)).to(tl.bfloat16)
        product = (silu_bf16.to(tl.float32) * up_acc).to(tl.bfloat16)
        tl.store(
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            product,
            mask=mask_m[:, None] & mask_n[None, :],
        )


def _dual_gate_up_eligible(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
) -> bool:
    if not triton_swiglu_available() or not _cuda_tensors_for_triton(x, w_gate, w_up):
        return False
    if x.dtype != torch.bfloat16 or w_gate.dtype != torch.bfloat16 or w_up.dtype != torch.bfloat16:
        return False
    dim = int(x.shape[-1])
    hidden = int(w_gate.shape[0])
    return (
        w_up.shape == w_gate.shape
        and hidden == 4 * dim
        and 256 <= dim <= 2048
    )


def _fused_gate_up(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    out: torch.Tensor,
) -> None:
    m, k = x.shape
    n = w_gate.shape[0]
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.cuda.device(x.device):
        _fused_gate_up_swiglu_kernel[grid](
            x,
            w_gate,
            w_up,
            out,
            m,
            k,
            n,
            x.stride(0),
            x.stride(1),
            w_gate.stride(0),
            w_gate.stride(1),
            w_up.stride(0),
            w_up.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )


def _fused_up_mul(
    x: torch.Tensor,
    w_up: torch.Tensor,
    silu_gate: torch.Tensor,
) -> None:
    m, k = x.shape
    n = w_up.shape[0]
    block_m, block_n, block_k = 64, 64, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.cuda.device(x.device):
        _fused_up_mul_kernel[grid](
            x,
            w_up,
            silu_gate,
            m,
            k,
            n,
            x.stride(0),
            x.stride(1),
            w_up.stride(0),
            w_up.stride(1),
            silu_gate.stride(0),
            silu_gate.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )


def residual_modulating_mlp_exact(
    x: torch.Tensor,
    mlp: Any,
    norm: Any,
    scale: torch.Tensor,
    shift: torch.Tensor,
    *,
    tile_size: int = DEFAULT_SWIGLU_TILE_SIZE,
) -> torch.Tensor:
    """In-place official ``x += swiglu(modulate(rms_norm(x)))``."""
    if x.numel() == 0:
        return x
    if not x.is_contiguous():
        x = x.contiguous()

    dim = int(x.shape[-1])
    x_flat = x.reshape(-1, dim)
    if not x_flat.is_contiguous():
        raise ValueError("Stage-5 x must be contiguous channels-last.")

    w_gate = mlp.w_gate.weight
    w_up = mlp.w_up.weight
    w_down = mlp.w_down.weight
    hidden = int(w_gate.shape[0])
    if tuple(w_up.shape) != tuple(w_gate.shape) or int(w_gate.shape[1]) != dim:
        raise ValueError(f"Incompatible SwiGLU gate/up shapes: {tuple(w_gate.shape)}, {tuple(w_up.shape)}.")
    if tuple(w_down.shape) != (dim, hidden):
        raise ValueError(f"Incompatible SwiGLU down shape: {tuple(w_down.shape)}.")

    s = scale.reshape(-1, dim)
    sh = shift.reshape(-1, dim)
    if s.shape != sh.shape or int(s.shape[0]) != 1:
        raise ValueError(
            "Official chunked Stage-5 MLP requires one broadcast channel affine; "
            f"got scale={tuple(scale.shape)}, shift={tuple(shift.shape)}."
        )

    max_chunk = min(max(1, int(tile_size)), int(x_flat.shape[0]))
    workspace = torch.empty((max_chunk, hidden), device=x.device, dtype=x.dtype)
    y_buf = torch.empty((max_chunk, dim), device=x.device, dtype=x.dtype)
    out_buf = torch.empty((max_chunk, dim), device=x.device, dtype=x.dtype)

    use_triton = bool(triton_swiglu_available() and _cuda_tensors_for_triton(x_flat, w_gate, w_up, w_down))
    use_dual = bool(use_triton and _dual_gate_up_eligible(x_flat, w_gate, w_up))
    w_gate_c = w_gate.contiguous() if use_dual else w_gate
    w_up_c = w_up.contiguous() if use_triton else w_up
    eps = float(getattr(norm, "eps", 1e-6))

    for start in range(0, int(x_flat.shape[0]), int(max_chunk)):
        end = min(start + int(max_chunk), int(x_flat.shape[0]))
        n = end - start
        xc = x_flat[start:end]
        y = y_buf[:n]
        y.copy_(F.rms_norm(xc, (dim,), norm.weight, eps))
        y.mul_(1.0 + s).add_(sh)

        ws = workspace[:n]
        if use_dual:
            _fused_gate_up(y, w_gate_c, w_up_c, ws)
        else:
            torch.mm(y, w_gate.t(), out=ws)
            F.silu(ws, inplace=True)
            if use_triton:
                _fused_up_mul(y, w_up_c, ws)
            else:
                ws.mul_(F.linear(y, w_up))
        torch.mm(ws, w_down.t(), out=out_buf[:n])
        xc.add_(out_buf[:n])
    return x

