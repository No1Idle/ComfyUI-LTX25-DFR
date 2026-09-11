# ruff: noqa: ANN001, ANN202, PLR0913
"""Official Triton joint video/keyframe 3-D neighborhood attention.

Ported from LTX-2.5 ``fallback_na/joint_triton.py``. One program owns a
block of queries along width and folds the local and cross-stream keys into
one online softmax without materializing the score tensor.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .decoder_joint_attention import (
    KEYFRAME_CONTEXT_SLOTS,
    keyframe_video_slots,
    video_keyframe_slots,
)


# Triton requires module-level JIT globals to be constexpr instances.
_NEG_INF = tl.constexpr(-3.0e38)
_LSE_FLOOR = tl.constexpr(1e-30)


@triton.jit
def _fold_w_run(
    m_i,
    l_i,
    acc,
    q_blk,
    k_ptr,
    v_ptr,
    plane_base,
    row_ok,
    w_lo,
    w_hi,
    w_start,
    w_end,
    s_w,
    d_off,
    d_mask,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    """Fold one row's W run into the running softmax state."""
    for wk0 in range(w_lo, w_hi, block_k):
        wk = wk0 + tl.arange(0, block_k)
        kmask = wk < w_hi
        kv_ptrs = plane_base + wk[:, None] * s_w + d_off[None, :]
        kv_mask = kmask[:, None] & d_mask[None, :] & row_ok
        k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
        k_t = tl.trans(k_blk)
        s = tl.dot(q_blk, k_t, input_precision="ieee") if is_fp32 else tl.dot(q_blk, k_t)
        vis = (
            (wk[None, :] >= w_start[:, None])
            & (wk[None, :] < w_end[:, None])
            & kmask[None, :]
            & row_ok
        )
        s = tl.where(vis, s, _NEG_INF)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        # A wholly masked row must contribute zero, not exp(0) == 1.
        p = tl.where(vis, tl.exp(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
        if is_fp32:
            acc = acc * alpha[:, None] + tl.dot(p, v_blk, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v_blk.dtype), v_blk)
        m_i = m_new
    return m_i, l_i, acc


@triton.jit
def _joint_video_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    kf_k_ptr,
    kf_v_ptr,
    slot_ptr,
    out_ptr,
    t_size,
    h_size,
    w_size,
    num_heads,
    bn_count,
    s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    kf_s_b,
    kf_s_p,
    kt: tl.constexpr,
    kh: tl.constexpr,
    kw: tl.constexpr,
    num_slots: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    """Video queries: local window plus ``num_slots`` keyframe planes."""
    pid_w = tl.program_id(0)
    h_q = tl.program_id(1)
    pid_tbn = tl.program_id(2)

    # Keep H on grid axis 1 and fold (T, B, NH) into axis 2. Pointer
    # arithmetic stays int64 because production temporal offsets exceed int32.
    t_q = pid_tbn // bn_count
    pid_bn = pid_tbn % bn_count
    batch_i = pid_bn // num_heads
    head_i = pid_bn % num_heads
    base = batch_i.to(tl.int64) * s_b + head_i.to(tl.int64) * s_n
    kf_base = batch_i.to(tl.int64) * kf_s_b + head_i.to(tl.int64) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = (
        q_ptr
        + base
        + t_q.to(tl.int64) * s_t
        + h_q.to(tl.int64) * s_h
        + w_off[:, None] * s_w
        + d_off[None, :]
    )
    q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

    w_q = tl.where(w_valid, w_off, 0)
    w_start = tl.maximum(w_q - kw // 2, 0)
    w_end = tl.minimum(w_q - kw // 2 + kw, w_size)
    blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
    w_lo = tl.maximum(pid_w * block_q - kw // 2, 0)
    w_hi = tl.minimum(blk_last - kw // 2 + kw, w_size)

    m_i = tl.full((block_q,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

    for d_t in range(kt):
        tk = t_q - kt // 2 + d_t
        t_ok = (tk >= 0) & (tk < t_size)
        tk_c = tl.maximum(tl.minimum(tk, t_size - 1), 0)
        for d_h in range(kh):
            hk = h_q - kh // 2 + d_h
            row_ok = t_ok & (hk >= 0) & (hk < h_size)
            hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
            m_i, l_i, acc = _fold_w_run(
                m_i,
                l_i,
                acc,
                q_blk,
                k_ptr,
                v_ptr,
                base + tk_c.to(tl.int64) * s_t + hk_c.to(tl.int64) * s_h,
                row_ok,
                w_lo,
                w_hi,
                w_start,
                w_end,
                s_w,
                d_off,
                d_mask,
                block_k=block_k,
                is_fp32=is_fp32,
            )

    # Cross-stream keys share the same online softmax state.
    for slot in range(num_slots):
        plane = tl.load(slot_ptr + t_q * num_slots + slot)
        plane_ok = plane >= 0
        plane_c = tl.maximum(plane, 0)
        for d_h in range(kh):
            hk = h_q - kh // 2 + d_h
            row_ok = plane_ok & (hk >= 0) & (hk < h_size)
            hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
            m_i, l_i, acc = _fold_w_run(
                m_i,
                l_i,
                acc,
                q_blk,
                kf_k_ptr,
                kf_v_ptr,
                kf_base + plane_c.to(tl.int64) * kf_s_p + hk_c.to(tl.int64) * s_h,
                row_ok,
                w_lo,
                w_hi,
                w_start,
                w_end,
                s_w,
                d_off,
                d_mask,
                block_k=block_k,
                is_fp32=is_fp32,
            )

    out = acc / tl.maximum(l_i, _LSE_FLOOR)[:, None]
    out_ptrs = (
        out_ptr
        + base
        + t_q.to(tl.int64) * s_t
        + h_q.to(tl.int64) * s_h
        + w_off[:, None] * s_w
        + d_off[None, :]
    )
    tl.store(
        out_ptrs,
        out.to(out_ptr.dtype.element_ty),
        mask=w_valid[:, None] & d_mask[None, :],
    )


@triton.jit
def _joint_keyframe_kernel(
    kf_q_ptr,
    kf_k_ptr,
    kf_v_ptr,
    k_ptr,
    v_ptr,
    slot_ptr,
    valid_ptr,
    out_ptr,
    h_size,
    w_size,
    num_heads,
    bn_count,
    s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    kf_s_b,
    kf_s_p,
    kh: tl.constexpr,
    kw: tl.constexpr,
    num_slots: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    """Keyframe queries: own plane plus ``num_slots`` video frames."""
    pid_w = tl.program_id(0)
    h_q = tl.program_id(1)
    pid_pbn = tl.program_id(2)

    p_q = pid_pbn // bn_count
    pid_bn = pid_pbn % bn_count
    batch_i = pid_bn // num_heads
    head_i = pid_bn % num_heads
    base = batch_i.to(tl.int64) * s_b + head_i.to(tl.int64) * s_n
    kf_base = batch_i.to(tl.int64) * kf_s_b + head_i.to(tl.int64) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = (
        kf_q_ptr
        + kf_base
        + p_q.to(tl.int64) * kf_s_p
        + h_q.to(tl.int64) * s_h
        + w_off[:, None] * s_w
        + d_off[None, :]
    )
    q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

    w_q = tl.where(w_valid, w_off, 0)
    w_start = tl.maximum(w_q - kw // 2, 0)
    w_end = tl.minimum(w_q - kw // 2 + kw, w_size)
    blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
    w_lo = tl.maximum(pid_w * block_q - kw // 2, 0)
    w_hi = tl.minimum(blk_last - kw // 2 + kw, w_size)

    m_i = tl.full((block_q,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

    own_ok = tl.load(valid_ptr + p_q) != 0
    for d_h in range(kh):
        hk = h_q - kh // 2 + d_h
        row_ok = own_ok & (hk >= 0) & (hk < h_size)
        hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
        m_i, l_i, acc = _fold_w_run(
            m_i,
            l_i,
            acc,
            q_blk,
            kf_k_ptr,
            kf_v_ptr,
            kf_base + p_q.to(tl.int64) * kf_s_p + hk_c.to(tl.int64) * s_h,
            row_ok,
            w_lo,
            w_hi,
            w_start,
            w_end,
            s_w,
            d_off,
            d_mask,
            block_k=block_k,
            is_fp32=is_fp32,
        )

    for slot in range(num_slots):
        frame = tl.load(slot_ptr + p_q * num_slots + slot)
        frame_ok = frame >= 0
        frame_c = tl.maximum(frame, 0)
        for d_h in range(kh):
            hk = h_q - kh // 2 + d_h
            row_ok = frame_ok & (hk >= 0) & (hk < h_size)
            hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
            m_i, l_i, acc = _fold_w_run(
                m_i,
                l_i,
                acc,
                q_blk,
                k_ptr,
                v_ptr,
                base + frame_c.to(tl.int64) * s_t + hk_c.to(tl.int64) * s_h,
                row_ok,
                w_lo,
                w_hi,
                w_start,
                w_end,
                s_w,
                d_off,
                d_mask,
                block_k=block_k,
                is_fp32=is_fp32,
            )

    out = acc / tl.maximum(l_i, _LSE_FLOOR)[:, None]
    out_ptrs = (
        out_ptr
        + kf_base
        + p_q.to(tl.int64) * kf_s_p
        + h_q.to(tl.int64) * s_h
        + w_off[:, None] * s_w
        + d_off[None, :]
    )
    tl.store(
        out_ptrs,
        out.to(out_ptr.dtype.element_ty),
        mask=w_valid[:, None] & d_mask[None, :],
    )


def _check_shared_layout(video: torch.Tensor, keyframes: torch.Tensor) -> None:
    if video.shape[2:] != keyframes.shape[2:]:
        raise ValueError(
            "video and keyframe streams must agree on H/W/NH/HD, got "
            f"{tuple(video.shape)} vs {tuple(keyframes.shape)}"
        )
    if not video.is_contiguous() or not keyframes.is_contiguous():
        raise ValueError("joint Triton NA needs both streams contiguous")
    if video.stride()[2:5] != keyframes.stride()[2:5]:
        raise ValueError(
            "H/W/NH strides must match across streams, got "
            f"{video.stride()[2:5]} vs {keyframes.stride()[2:5]}"
        )


def joint_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: tuple[int, int, int],
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint attention with the official flash-style Triton kernels."""
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    keyframe_q = keyframe_q.contiguous()
    keyframe_k = keyframe_k.contiguous()
    keyframe_v = keyframe_v.contiguous()
    _check_shared_layout(q, keyframe_q)

    batch, time, height, width, heads, head_dim = q.shape
    planes = keyframe_q.shape[1]
    if keyframe_times.shape != (planes,) or keyframe_valid.shape != (planes,):
        raise ValueError(
            f"keyframe_times/keyframe_valid must be ({planes},), got "
            f"{tuple(keyframe_times.shape)} / {tuple(keyframe_valid.shape)}"
        )
    kernel_t, kernel_h, kernel_w = kernel_size

    video_slots = video_keyframe_slots(keyframe_times, keyframe_valid, time, num_slots)
    keyframe_slots = keyframe_video_slots(keyframe_times, keyframe_valid, time, num_slots)
    video_slots = video_slots.to(device=q.device, dtype=torch.int32).contiguous()
    keyframe_slots = keyframe_slots.to(device=q.device, dtype=torch.int32).contiguous()
    valid_i32 = keyframe_valid.to(device=q.device, dtype=torch.int32).contiguous()

    video_out = torch.empty_like(q)
    keyframe_out = torch.empty_like(keyframe_q)

    hd_pad = max(16, triton.next_power_of_2(head_dim))
    block_q = 16
    # RTX 4070 Ti benchmark: ~12% lower joint-attention time with identical
    # tested BF16 outputs. Keep the original launch for other formats/devices.
    num_warps = 2 if (
        q.dtype == torch.bfloat16
        and head_dim == 64
        and (kernel_t, kernel_h, kernel_w) == (11, 11, 11)
        and torch.cuda.get_device_capability(q.device) == (8, 9)
    ) else 4
    block_k = max(16, min(32, triton.next_power_of_2(min(width, block_q + kernel_w))))
    is_fp32 = q.dtype == torch.float32
    strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3), q.stride(4))
    kf_strides = (keyframe_q.stride(0), keyframe_q.stride(1))

    bn_count = batch * heads
    _joint_video_kernel[(triton.cdiv(width, block_q), height, time * bn_count)](
        q,
        k,
        v,
        keyframe_k,
        keyframe_v,
        video_slots,
        video_out,
        time,
        height,
        width,
        heads,
        bn_count,
        *strides,
        *kf_strides,
        kt=kernel_t,
        kh=kernel_h,
        kw=kernel_w,
        num_slots=num_slots,
        hd=head_dim,
        hd_pad=hd_pad,
        block_q=block_q,
        block_k=block_k,
        is_fp32=is_fp32,
        num_warps=num_warps,
    )
    _joint_keyframe_kernel[(triton.cdiv(width, block_q), height, planes * bn_count)](
        keyframe_q,
        keyframe_k,
        keyframe_v,
        k,
        v,
        keyframe_slots,
        valid_i32,
        keyframe_out,
        height,
        width,
        heads,
        bn_count,
        *strides,
        *kf_strides,
        kh=kernel_h,
        kw=kernel_w,
        num_slots=num_slots,
        hd=head_dim,
        hd_pad=hd_pad,
        block_q=block_q,
        block_k=block_k,
        is_fp32=is_fp32,
        num_warps=num_warps,
    )
    return video_out, keyframe_out
