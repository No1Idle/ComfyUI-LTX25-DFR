"""Official LTX-2.5 CHUNKED_EAGER Stage-5 dual-stream executor.

The Stage-4 pixel shuffle is deliberately deferred.  Every Stage-5 block
injects the low-resolution Stage-4 feature through four width chunks, runs
joint video/keyframe neighborhood attention in four fixed-extent halo slabs,
then executes the official 16,384-token tiled SwiGLU residual.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from einops import rearrange

from .decoder_det_dual_stream_exact import _forward_attention_with_keyframes_exact
from .decoder_stage5_swiglu_chunked import residual_modulating_mlp_exact


OFFICIAL_STAGE5_W_CHUNKS = 4


def stage5_grid_shape_from_stage4(
    stage4_feat: torch.Tensor,
    stride: tuple[int, int, int],
    *,
    drop_leading_frame: bool,
) -> tuple[int, int, int]:
    t, h, w = (int(v) for v in stage4_feat.shape[1:4])
    st, sh, sw = (int(v) for v in stride)
    return (
        t * st - (1 if drop_leading_frame and st == 2 else 0),
        h * sh,
        w * sw,
    )


def _upsample_then_context_exact(
    feat: torch.Tensor,
    upsample: Any,
    context_proj: Any,
    *,
    drop_leading_frame: bool,
) -> torch.Tensor:
    """Official ``context_proj(pixel_shuffle(upsample.proj(feat)))``."""
    p1, p2, p3 = (int(v) for v in upsample.stride)
    up = F.linear(feat, upsample.proj.weight, upsample.proj.bias)
    up = rearrange(
        up,
        "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c",
        p1=p1,
        p2=p2,
        p3=p3,
    )
    if p1 == 2 and drop_leading_frame:
        up = up[:, 1:, :, :, :]
    return F.linear(up, context_proj.weight, context_proj.bias)


def inject_deferred_context_exact(
    x: torch.Tensor,
    stage4_feat: torch.Tensor,
    upsample: Any,
    context_proj: Any,
    *,
    w_chunks: int = OFFICIAL_STAGE5_W_CHUNKS,
    drop_leading_frame: bool,
) -> torch.Tensor:
    """Official eager four-way deferred Stage-4 context injection."""
    if int(w_chunks) <= 1:
        x.add_(
            _upsample_then_context_exact(
                stage4_feat,
                upsample,
                context_proj,
                drop_leading_frame=drop_leading_frame,
            )
        )
        return x

    p3 = int(upsample.stride[2])
    w_hi = int(x.shape[3])
    lo = 0
    for feat_chunk in torch.chunk(stage4_feat, int(w_chunks), dim=3):
        hi = min(w_hi, lo + int(feat_chunk.shape[3]) * p3)
        ctx = _upsample_then_context_exact(
            feat_chunk,
            upsample,
            context_proj,
            drop_leading_frame=drop_leading_frame,
        )
        x[:, :, :, lo:hi, :].add_(ctx[:, :, :, : hi - lo, :])
        lo = hi
    if lo != w_hi:
        raise RuntimeError(f"Deferred context injection filled W={lo}, expected W={w_hi}.")
    return x


def inject_deferred_keyframe_context_exact(
    x: torch.Tensor,
    stage4_feat: torch.Tensor,
    upsample: Any,
    context_proj: Any,
    *,
    w_chunks: int = OFFICIAL_STAGE5_W_CHUNKS,
) -> torch.Tensor:
    """Keyframe planes use the same inject with planes folded into batch."""
    if x.shape[:2] != stage4_feat.shape[:2]:
        raise ValueError(
            f"Keyframe x/Stage-4 features disagree on (B,P): {tuple(x.shape[:2])} vs "
            f"{tuple(stage4_feat.shape[:2])}."
        )
    if not x.is_contiguous():
        raise ValueError("Keyframe Stage-5 x must be contiguous before deferred injection.")
    feat = stage4_feat.contiguous()
    folded = int(x.shape[0] * x.shape[1])
    inject_deferred_context_exact(
        x.reshape(folded, 1, *x.shape[2:]),
        feat.reshape(folded, 1, *feat.shape[2:]),
        upsample,
        context_proj,
        w_chunks=int(w_chunks),
        drop_leading_frame=True,
    )
    return x


def build_deferred_stage5_inputs_exact(
    comfy_na: Any,
    decoder: Any,
    stage4_feat: torch.Tensor,
    keyframe_stage4_feat: torch.Tensor,
    video_x_t: torch.Tensor,
    keyframe_x_t: torch.Tensor,
    keyframe_valid: torch.Tensor,
    *,
    drop_leading_frame: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official ``_x_for_diff_step`` and keyframe counterpart (no context concat)."""
    patched = comfy_na.patchify(video_x_t, patch_size_hw=int(decoder.patch_size), patch_size_t=1)
    video_x = decoder.conv_in_x_t(patched.permute(0, 2, 3, 4, 1)).contiguous()

    key_patched = comfy_na.patchify(
        keyframe_x_t,
        patch_size_hw=int(decoder.patch_size),
        patch_size_t=1,
    )
    keyframe_x = decoder.conv_in_x_t(key_patched.permute(0, 2, 3, 4, 1))
    keyframe_x = (
        keyframe_x
        * keyframe_valid.to(device=keyframe_x.device, dtype=keyframe_x.dtype)[None, :, None, None, None]
    ).contiguous()

    expected_video = stage5_grid_shape_from_stage4(
        stage4_feat,
        tuple(int(v) for v in decoder.upsamples[3].stride),
        drop_leading_frame=bool(drop_leading_frame),
    )
    if tuple(int(v) for v in video_x.shape[1:4]) != expected_video:
        raise ValueError(
            f"Deferred Stage-5 video geometry mismatch: x={tuple(video_x.shape)}, "
            f"expected grid={expected_video}."
        )
    expected_keyframe = (
        int(keyframe_stage4_feat.shape[1]),
        int(keyframe_stage4_feat.shape[2]) * int(decoder.upsamples[3].stride[1]),
        int(keyframe_stage4_feat.shape[3]) * int(decoder.upsamples[3].stride[2]),
    )
    if tuple(int(v) for v in keyframe_x.shape[1:4]) != expected_keyframe:
        raise ValueError(
            f"Deferred Stage-5 keyframe geometry mismatch: x={tuple(keyframe_x.shape)}, "
            f"expected grid={expected_keyframe}."
        )
    return video_x, keyframe_x


def cut_w_slab_exact(
    x: torch.Tensor,
    core_start: int,
    core_end: int,
    halo: int,
    extent: int,
    left_halo: torch.Tensor | None,
    *,
    has_right_neighbor: bool,
) -> torch.Tensor:
    """Official fixed-extent core + neighbor halos + replicated true edges."""
    width = int(x.shape[3])
    channels = int(x.shape[4])
    buf = x.new_zeros(*x.shape[:3], int(extent), channels)
    core_len = int(core_end - core_start)
    if left_halo is not None:
        left_len = int(left_halo.shape[3])
        buf[:, :, :, halo - left_len : halo, :] = left_halo
    buf[:, :, :, halo : halo + core_len, :] = x[:, :, :, core_start:core_end, :]

    right_filled = 0
    if has_right_neighbor:
        right_end = min(width, core_end + halo)
        right = x[:, :, :, core_end:right_end, :]
        right_filled = int(right.shape[3])
        buf[:, :, :, halo + core_len : halo + core_len + right_filled, :] = right

    if left_halo is None and halo > 0 and core_len > 0:
        edge_l = buf[:, :, :, halo : halo + 1, :]
        buf[:, :, :, :halo, :] = edge_l.expand(*x.shape[:3], halo, channels)
    missing_right = extent - (halo + core_len + right_filled)
    if missing_right > 0 and core_len > 0:
        edge_r = buf[:, :, :, halo + core_len - 1 : halo + core_len, :]
        lo_r = halo + core_len + right_filled
        buf[:, :, :, lo_r:extent, :] = edge_r.expand(*x.shape[:3], missing_right, channels)
    return buf


def w_slab_geometry_exact(width: int, w_chunks: int, halo: int) -> tuple[int, int]:
    chunk_w = (int(width) + int(w_chunks) - 1) // int(w_chunks)
    return chunk_w, chunk_w + 2 * int(halo)


def run_w_chunked_joint_residual_exact(
    x: torch.Tensor,
    keyframe_x: torch.Tensor,
    *,
    w_chunks: int,
    halo: int,
    attention_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official in-place dual-stream four-slab attention residual loop."""
    if x.shape[2:4] != keyframe_x.shape[2:4]:
        raise ValueError(
            f"Chunked joint attention needs matching H/W: {tuple(x.shape[2:4])} vs "
            f"{tuple(keyframe_x.shape[2:4])}."
        )
    width = int(x.shape[3])
    chunk_w, extent = w_slab_geometry_exact(width, int(w_chunks), int(halo))
    left_halo: torch.Tensor | None = None
    keyframe_left_halo: torch.Tensor | None = None

    for i in range(int(w_chunks)):
        core_start = i * chunk_w
        core_end = min(width, (i + 1) * chunk_w)
        core_len = core_end - core_start
        if core_len <= 0:
            continue
        has_right = i + 1 < int(w_chunks) and core_end < width
        buf = cut_w_slab_exact(
            x,
            core_start,
            core_end,
            int(halo),
            int(extent),
            left_halo,
            has_right_neighbor=has_right,
        )
        keyframe_buf = cut_w_slab_exact(
            keyframe_x,
            core_start,
            core_end,
            int(halo),
            int(extent),
            keyframe_left_halo,
            has_right_neighbor=has_right,
        )
        if has_right:
            take = min(int(halo), int(core_len))
            left_halo = x[:, :, :, core_end - take : core_end, :].clone()
            keyframe_left_halo = keyframe_x[:, :, :, core_end - take : core_end, :].clone()

        w_pos = torch.arange(extent, device=x.device, dtype=torch.float32) + (core_start - halo)
        out, keyframe_out = attention_fn(buf, keyframe_buf, w_pos)
        x[:, :, :, core_start:core_end, :].add_(
            out[:, :, :, halo : halo + core_len, :]
        )
        keyframe_x[:, :, :, core_start:core_end, :].add_(
            keyframe_out[:, :, :, halo : halo + core_len, :]
        )
    return x, keyframe_x


def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


def forward_stage5_chunked_block_exact(
    comfy_na: Any,
    comfy_kitchen: Any,
    block: Any,
    upsample: Any,
    video_x: torch.Tensor,
    video_stage4_feat: torch.Tensor,
    keyframe_x: torch.Tensor,
    keyframe_stage4_feat: torch.Tensor,
    modulation: tuple[torch.Tensor, ...],
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    *,
    drop_leading_frame: bool,
    num_slots: int,
    w_chunks: int = OFFICIAL_STAGE5_W_CHUNKS,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(modulation) != 7:
        raise ValueError(f"Expected 7 AdaLN chunks, got {len(modulation)}.")
    table = block.scale_shift_table
    chunks = [modulation[i] + table[i].view(1, 1, 1, 1, -1) for i in range(7)]
    scale_msa, shift_msa, _gate_msa, scale_mlp, shift_mlp, _gate_mlp, _extra = chunks

    inject_deferred_context_exact(
        video_x,
        video_stage4_feat,
        upsample,
        block.context_proj,
        w_chunks=int(w_chunks),
        drop_leading_frame=bool(drop_leading_frame),
    )
    inject_deferred_keyframe_context_exact(
        keyframe_x,
        keyframe_stage4_feat,
        upsample,
        block.context_proj,
        w_chunks=int(w_chunks),
    )

    def attend(
        video_slab: torch.Tensor,
        keyframe_slab: torch.Tensor,
        width_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_y = _modulate(block.norm1(video_slab), scale_msa, shift_msa)
        keyframe_y = _modulate(block.norm1(keyframe_slab), scale_msa, shift_msa)
        return _forward_attention_with_keyframes_exact(
            comfy_na,
            comfy_kitchen,
            block.attn,
            video_y,
            keyframe_y,
            keyframe_times,
            keyframe_valid,
            num_slots=int(num_slots),
            width_positions=width_positions,
            separate_qkv=True,
        )

    video_x, keyframe_x = run_w_chunked_joint_residual_exact(
        video_x,
        keyframe_x,
        w_chunks=int(w_chunks),
        halo=int(block.attn.kernel_size[2]) // 2,
        attention_fn=attend,
    )
    video_x = residual_modulating_mlp_exact(
        video_x,
        block.mlp,
        block.norm2,
        scale_mlp,
        shift_mlp,
    )
    keyframe_x = residual_modulating_mlp_exact(
        keyframe_x,
        block.mlp,
        block.norm2,
        scale_mlp,
        shift_mlp,
    )
    keyframe_x.mul_(
        keyframe_valid.to(device=keyframe_x.device, dtype=keyframe_x.dtype)[None, :, None, None, None]
    )
    return video_x, keyframe_x


def _pixels_from_stage5(comfy_na: Any, decoder: Any, x: torch.Tensor) -> torch.Tensor:
    x = decoder.norm_out(x)
    x = decoder.conv_out(x)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return comfy_na.unpatchify(x, patch_size_hw=int(decoder.patch_size), patch_size_t=1)


def forward_stage5_chunked_step_exact(
    comfy_na: Any,
    comfy_kitchen: Any,
    decoder: Any,
    video_x: torch.Tensor,
    video_stage4_feat: torch.Tensor,
    keyframe_x: torch.Tensor,
    keyframe_stage4_feat: torch.Tensor,
    t: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    *,
    drop_leading_frame: bool,
    num_slots: int,
    w_chunks: int = OFFICIAL_STAGE5_W_CHUNKS,
    collect_diagnostics: bool = True,
    return_keyframes: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, float, int]:
    """Official ``forward_diff_step_deferred_with_keyframes`` on Comfy weights."""
    video_x = video_x.contiguous()
    keyframe_x = keyframe_x.contiguous()
    video_stage4_feat = video_stage4_feat.contiguous()
    keyframe_stage4_feat = keyframe_stage4_feat.contiguous()

    t_emb = decoder.t_embedder(
        float(decoder.timestep_scale_multiplier) * t,
        dtype=video_x.dtype,
    )
    modulation = decoder.shared_adaln(t_emb)
    invalid_zero = 0.0
    blocks_executed = 0
    upsample = decoder.upsamples[3]
    for block in decoder.diff_blocks:
        video_x, keyframe_x = forward_stage5_chunked_block_exact(
            comfy_na,
            comfy_kitchen,
            block,
            upsample,
            video_x,
            video_stage4_feat,
            keyframe_x,
            keyframe_stage4_feat,
            modulation,
            keyframe_times,
            keyframe_valid,
            drop_leading_frame=bool(drop_leading_frame),
            num_slots=int(num_slots),
            w_chunks=int(w_chunks),
        )
        blocks_executed += 1
        if collect_diagnostics and bool((~keyframe_valid).any()):
            invalid = keyframe_x[:, ~keyframe_valid]
            if invalid.numel():
                invalid_zero = max(invalid_zero, float(invalid.float().abs().max().item()))

    keyframe_pixels = _pixels_from_stage5(comfy_na, decoder, keyframe_x) if return_keyframes else None
    return (
        _pixels_from_stage5(comfy_na, decoder, video_x),
        keyframe_pixels,
        float(invalid_zero),
        int(blocks_executed),
    )
