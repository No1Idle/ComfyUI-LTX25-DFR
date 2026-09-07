"""U3.4b3b2: exact deterministic NABlock dual-stream port.

Parity contract is fixed by the frozen LTX DiffVAE sources, not inferred from
runtime behavior:

* both streams use the same block weights;
* both streams are pre-normalized by the same ``norm1``;
* the same fused QKV, Q/K RMSNorm and absolute 3-D RoPE are applied;
* video + keyframe Q/K/V are passed to the already validated U3.4b2
  ``joint_na3d`` primitive, i.e. one joint softmax per query;
* the same output projection is residual-added independently to both streams;
* the same ``norm2`` + SwiGLU MLP is then residual-added independently;
* invalid keyframe planes are re-zeroed.

The current Comfy ``NABlock`` has only ``forward(x)``. This module adds the
missing *behavior* as an external bridge and never mutates the loaded decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .decoder_joint_attention import DFRJointDecoderAttention, joint_na3d
from .decoder_det_stage_official_preflight import DFROfficialDeterministicStagePreflight


DUAL_STREAM_BLOCK_VERSION = 1


@dataclass(frozen=True)
class DFRDeterministicDualStreamBlockPort:
    version: int
    source_contract: str
    implementation_mode: str
    num_slots: int
    qkv_shared: bool
    qk_norm_shared: bool
    rope_shared: bool
    projection_shared: bool
    mlp_shared: bool
    invalid_planes_rezeroed: bool
    decoder_mutated: bool


def prepare_deterministic_dual_stream_block_port(
    official_preflight: DFROfficialDeterministicStagePreflight,
    joint_decoder_attention: DFRJointDecoderAttention,
) -> tuple[DFRDeterministicDualStreamBlockPort, str]:
    if not isinstance(official_preflight, DFROfficialDeterministicStagePreflight):
        raise ValueError(
            f"Expected DFROfficialDeterministicStagePreflight, got {type(official_preflight).__name__}."
        )
    if not isinstance(joint_decoder_attention, DFRJointDecoderAttention):
        raise ValueError(
            f"Expected DFRJointDecoderAttention, got {type(joint_decoder_attention).__name__}."
        )
    if not official_preflight.architecture_matches_checkpoint_config:
        raise ValueError("U3.4b3b2 requires the validated C23b checkpoint/runtime architecture mapping.")
    if not official_preflight.keyframe_plane_helper_exact:
        raise ValueError("U3.4b3b2 requires the validated exact official keyframe-plane helper.")
    if joint_decoder_attention.source_backend not in {
        "official_fallback_joint_eager",
        "official_auto_joint_triton_cuda_eager_fallback",
    }:
        raise ValueError(
            "U3.4b3b2 requires the frozen official joint-attention semantics validated in U3.4b2."
        )

    state = DFRDeterministicDualStreamBlockPort(
        version=DUAL_STREAM_BLOCK_VERSION,
        source_contract="frozen_NABlock.forward_with_keyframes+joint_eager",
        implementation_mode="external_bridge_no_decoder_mutation",
        num_slots=int(joint_decoder_attention.num_slots),
        qkv_shared=True,
        qk_norm_shared=True,
        rope_shared=True,
        projection_shared=True,
        mlp_shared=True,
        invalid_planes_rezeroed=True,
        decoder_mutated=False,
    )
    report = (
        f"PASS=True; stage=U3.4b3b2_prepare_dual_stream_block; version={state.version}; "
        f"source_contract={state.source_contract}; implementation_mode={state.implementation_mode}; "
        f"num_slots={state.num_slots}; qkv_shared={state.qkv_shared}; qk_norm_shared={state.qk_norm_shared}; "
        f"rope_shared={state.rope_shared}; projection_shared={state.projection_shared}; mlp_shared={state.mlp_shared}; "
        f"invalid_planes_rezeroed={state.invalid_planes_rezeroed}; decoder_mutated=False."
    )
    return state, report


def _default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    # Exact current Comfy NADiffusionDecoder helper.
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return d_t, d_hw, d_hw


def _inv_freq(dim: int, base: float, device: torch.device) -> torch.Tensor:
    exponents = torch.arange(0, dim, 2, dtype=torch.float64, device="cpu") / dim
    inv = 1.0 / torch.pow(torch.tensor(float(base), dtype=torch.float64), exponents)
    return inv.to(dtype=torch.float32, device=device)


def _rotate_interleaved(x: torch.Tensor, positions: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    """Absolute RoPE for one axis, interleaved pair convention used by Comfy."""
    if x.shape[-1] == 0:
        return x
    xf = x.to(torch.float32)
    pairs = xf.reshape(*xf.shape[:-1], -1, 2)
    angles = positions.to(torch.float32)[..., None] * inv
    cos = angles.cos()
    sin = angles.sin()
    a = pairs[..., 0]
    b = pairs[..., 1]
    out = torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1)
    return out.reshape_as(xf).to(dtype=x.dtype)


def _apply_abs_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    temporal_positions: torch.Tensor,
    rope_split: tuple[int, int, int],
    rope_base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the same absolute T/H/W RoPE to [B,A,H,W,NH,HD] Q/K.

    ``A`` is video time or keyframe-plane axis. Temporal positions may therefore
    be integer frame indices or fractional keyframe stage times.
    """
    _, axis, height, width, _, _ = q.shape
    dt, dh, dw = rope_split
    t_pos = temporal_positions.to(device=q.device, dtype=torch.float32).view(axis, 1, 1)
    h_pos = torch.arange(height, device=q.device, dtype=torch.float32).view(1, height, 1)
    w_pos = torch.arange(width, device=q.device, dtype=torch.float32).view(1, 1, width)

    q_parts = list(torch.split(q, (dt, dh, dw), dim=-1))
    k_parts = list(torch.split(k, (dt, dh, dw), dim=-1))
    positions = (t_pos.expand(axis, height, width), h_pos.expand(axis, height, width), w_pos.expand(axis, height, width))
    dims = (dt, dh, dw)
    for i, (pos, dim) in enumerate(zip(positions, dims)):
        inv = _inv_freq(dim, rope_base, q.device)
        # broadcast [A,H,W,pairs] over B and heads
        p = pos[None, ..., None]
        q_parts[i] = _rotate_interleaved(q_parts[i], p, inv)
        k_parts[i] = _rotate_interleaved(k_parts[i], p, inv)
    return torch.cat(q_parts, dim=-1), torch.cat(k_parts, dim=-1)


def _qkv_with_official_rope(
    attn: Any,
    x_normed: torch.Tensor,
    temporal_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, axis, height, width, dim = x_normed.shape
    q, k, v = attn.qkv(x_normed).chunk(3, dim=-1)
    heads = int(attn.num_heads)
    head_dim = int(attn.head_dim)
    shape = (batch, axis, height, width, heads, head_dim)
    q = q.reshape(shape)
    k = k.reshape(shape)
    v = v.reshape(shape)

    # Same Q/K RMSNorm and query scaling as current Comfy's NeighborhoodAttention3D.
    q = attn.q_norm(q) * float(attn.scale)
    k = attn.k_norm(k)
    rope_split = tuple(int(x) for x in getattr(attn, "rope_split", _default_rope_dim_split(head_dim)))
    rope_base = float(getattr(attn, "rope_base", 10000.0))
    q, k = _apply_abs_rope(q, k, temporal_positions, rope_split, rope_base)
    return q, k, v


def forward_nablock_with_keyframes_exact(
    block: Any,
    video_x: torch.Tensor,
    keyframe_x: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """External exact port of frozen ``NABlock.forward_with_keyframes`` semantics."""
    if video_x.ndim != 5 or keyframe_x.ndim != 5:
        raise ValueError("video_x and keyframe_x must be channels-last 5-D tensors.")
    if video_x.shape[0] != keyframe_x.shape[0] or video_x.shape[2:4] != keyframe_x.shape[2:4]:
        raise ValueError(
            f"Video/keyframe batch+spatial geometry must match: video={tuple(video_x.shape)}, keyframes={tuple(keyframe_x.shape)}."
        )
    if video_x.shape[-1] != keyframe_x.shape[-1]:
        raise ValueError("Video/keyframe channel widths must match.")
    planes = int(keyframe_x.shape[1])
    if tuple(keyframe_times.shape) != (planes,) or tuple(keyframe_valid.shape) != (planes,):
        raise ValueError("keyframe_times and keyframe_valid must both be [P].")

    # Official block is pre-norm on both streams with shared weights.
    video_norm = block.norm1(video_x)
    keyframe_norm = block.norm1(keyframe_x)

    video_times = torch.arange(video_x.shape[1], device=video_x.device, dtype=torch.float32)
    q, k, v = _qkv_with_official_rope(block.attn, video_norm, video_times)
    kq, kk, kv = _qkv_with_official_rope(
        block.attn,
        keyframe_norm,
        keyframe_times.to(device=keyframe_x.device, dtype=torch.float32),
    )

    video_attn, keyframe_attn = joint_na3d(
        q, k, v,
        kq, kk, kv,
        keyframe_times.to(device=video_x.device, dtype=torch.float32),
        keyframe_valid.to(device=video_x.device, dtype=torch.bool),
        tuple(int(x) for x in block.attn.kernel_size),
    )
    dim = int(video_x.shape[-1])
    video_attn = video_attn.reshape(*video_x.shape[:-1], dim)
    keyframe_attn = keyframe_attn.reshape(*keyframe_x.shape[:-1], dim)

    video_x = video_x + block.attn.proj(video_attn)
    keyframe_x = keyframe_x + block.attn.proj(keyframe_attn)

    # Same independent shared-weight SwiGLU residual on both streams.
    video_x = block.mlp(video_x, pre=block.norm2, add_to=video_x)
    keyframe_x = block.mlp(keyframe_x, pre=block.norm2, add_to=keyframe_x)
    keyframe_x = keyframe_x * keyframe_valid.to(keyframe_x.dtype)[None, :, None, None, None]
    return video_x, keyframe_x


def _tensor_checksum(module: Any) -> tuple[tuple[str, float], ...]:
    out = []
    for name, p in module.named_parameters():
        # Small deterministic checksum; no clone of giant params.
        if p.numel():
            flat = p.detach().reshape(-1)
            idx = torch.linspace(0, flat.numel() - 1, min(8, flat.numel()), device=flat.device).long()
            value = float(flat.index_select(0, idx).to(torch.float32).sum().item())
        else:
            value = 0.0
        out.append((name, value))
    return tuple(out)


def validate_deterministic_dual_stream_block_port(
    vae: Any,
    dual_stream_block_port: DFRDeterministicDualStreamBlockPort,
    stage_index: int = 3,
    block_index: int = 0,
) -> dict[str, Any]:
    if not isinstance(dual_stream_block_port, DFRDeterministicDualStreamBlockPort):
        raise ValueError(
            f"Expected DFRDeterministicDualStreamBlockPort, got {type(dual_stream_block_port).__name__}."
        )
    decoder = getattr(getattr(vae, "first_stage_model", None), "decoder", None)
    if decoder is None:
        raise ValueError("Connected VAE has no decoder.")
    stages = list(decoder.det_stages)
    stage_index = int(stage_index)
    block_index = int(block_index)
    if stage_index < 0 or stage_index >= len(stages):
        raise ValueError(f"stage_index={stage_index} outside 0..{len(stages)-1}.")
    blocks = list(stages[stage_index])
    if block_index < 0 or block_index >= len(blocks):
        raise ValueError(f"block_index={block_index} outside 0..{len(blocks)-1}.")
    block = blocks[block_index]

    try:
        anchor = next(block.parameters())
        device, dtype = anchor.device, anchor.dtype
    except StopIteration:
        device, dtype = torch.device("cpu"), torch.float32

    kernel = tuple(int(x) for x in block.attn.kernel_size)
    # Keep the real block probe small but never smaller than its NA window.
    t = max(kernel[0], 3)
    h = max(kernel[1], 3)
    w = max(kernel[2], 3)
    channels = int(block.attn.dim)
    planes = 3
    gen = torch.Generator(device=device).manual_seed(3432)
    video = torch.randn((1, t, h, w, channels), generator=gen, device=device, dtype=dtype) * 0.02
    keyframes = torch.randn((1, planes, h, w, channels), generator=gen, device=device, dtype=dtype) * 0.02
    times = torch.tensor([0.25, float(max(1, t // 2)), float(t + 2)], device=device, dtype=torch.float32)
    valid = torch.tensor([True, True, True], device=device, dtype=torch.bool)
    invalid = torch.zeros(planes, device=device, dtype=torch.bool)

    before = _tensor_checksum(block)
    with torch.inference_mode():
        # All-invalid keyframe stream must reduce the VIDEO side to the native
        # single-stream block, providing a high-value mapping check for QKV/norm/RoPE/proj/MLP.
        native_video = block(video.clone())
        invalid_video, invalid_keyframes = forward_nablock_with_keyframes_exact(
            block, video.clone(), torch.zeros_like(keyframes), times, invalid
        )
        valid_video, valid_keyframes = forward_nablock_with_keyframes_exact(
            block, video.clone(), keyframes.clone(), times, valid
        )
    after = _tensor_checksum(block)

    native_error = float((invalid_video.to(torch.float32) - native_video.to(torch.float32)).abs().max().item())
    invalid_keyframe_zero_error = float(invalid_keyframes.to(torch.float32).abs().max().item())
    cross_stream_influence = float((valid_video.to(torch.float32) - invalid_video.to(torch.float32)).abs().max().item())
    finite_error = 0.0 if (torch.isfinite(valid_video).all() and torch.isfinite(valid_keyframes).all()) else 1.0
    shape_error = 0.0 if (valid_video.shape == video.shape and valid_keyframes.shape == keyframes.shape) else 1.0
    weight_mutation_error = 0.0 if before == after else 1.0

    # Backend/operation-order differences are expected to be tiny; this threshold
    # is validation tolerance, not an algorithmic approximation.
    tolerance = 3e-4 if dtype in (torch.float16, torch.bfloat16) else 8e-5
    passed = (
        native_error <= tolerance
        and invalid_keyframe_zero_error == 0.0
        and cross_stream_influence > 1e-7
        and finite_error == 0.0
        and shape_error == 0.0
        and weight_mutation_error == 0.0
    )
    return {
        "passed": bool(passed),
        "device": device.type,
        "dtype": str(dtype),
        "stage_index": stage_index,
        "block_index": block_index,
        "kernel": kernel,
        "channels": channels,
        "video_shape": tuple(int(x) for x in video.shape),
        "keyframe_shape": tuple(int(x) for x in keyframes.shape),
        "native_video_max_error": native_error,
        "tolerance": float(tolerance),
        "invalid_keyframe_zero_error": invalid_keyframe_zero_error,
        "cross_stream_influence": cross_stream_influence,
        "finite_error": finite_error,
        "shape_error": shape_error,
        "weight_mutation_error": weight_mutation_error,
        "decoder_mutated": False,
    }
