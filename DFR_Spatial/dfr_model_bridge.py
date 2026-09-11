"""Materialize official LTX DFR token state into Comfy LTX model-call geometry.

This is Phase B / sampler bridge sub-step 2.  It does *not* run the model.

Why this adapter exists
-----------------------
The official Lightricks transformer consumes a patchified ``LatentState``:
    latent        [B, tokens, C]
    denoise_mask  [B, tokens, 1]
    positions     [B, 3, tokens, 2]  (time already divided by fps)

Comfy's LTXV model consumes a 5-D latent and internally patchifies it.  Appended
conditioning/generated/reference groups are represented as extra latent frames,
with ``keyframe_idxs`` overriding their pixel coordinates.  Sparse lower-resolution
IC-LoRA references are embedded into a full target grid and holes are marked with
negative denoise-mask values so Comfy filters them before the transformer.

This module deterministically translates between those representations and exposes
an independent validator that emulates Comfy's input filtering/coordinate override
without importing ComfyUI.

Reference behavior checked against current ComfyUI:
    comfy/ldm/lightricks/model.py :: LTXVModel._process_input
    comfy/model_base.py           :: LTXV.process_timestep

Official semantics checked against Lightricks:
    ltx_pipelines.utils.helpers.modality_from_latent_state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .dfr_noiser import NOISER_METADATA_KEY
from .latent_state import get_official_state
from .frame_rate import official_conditioning_fps, model_video_positions


BRIDGE_VERSION = 1


def materialize_video_tokens_with_layout(tokens: torch.Tensor, layout: "DFRComfyModelInput") -> torch.Tensor:
    """Repack changing tokens using already-validated, immutable loop geometry.

    Masks, positions and operation metadata are built once by the full bridge.
    Sparse reference holes remain zero, exactly as in that bridge.
    """
    b, c, _, h, w = layout.x.shape
    expected = layout.total_tokens_after_filter
    if tuple(tokens.shape) != (b, expected, c):
        raise ValueError(f"Token geometry changed: {tuple(tokens.shape)}, expected {(b, expected, c)}.")
    cursor = layout.base_frames * h * w
    groups = [_unpatchify_tokens(tokens[:, :cursor], layout.base_frames, h, w)]
    for op in layout.operation_frames:
        frames = int(op["materialized_frames"])
        sh, sw = int(op["source_height"]), int(op["source_width"])
        count = frames * sh * sw
        group = _unpatchify_tokens(tokens[:, cursor:cursor + count], frames, sh, sw)
        cursor += count
        if op["sparse"]:
            expanded = tokens.new_zeros((b, c, frames, h, w))
            expanded[..., ::h // sh, ::w // sw] = group
            group = expanded
        groups.append(group)
    if cursor != expected:
        raise ValueError("Cached operation layout did not consume all video tokens.")
    return torch.cat(groups, dim=2)


@dataclass
class DFRComfyModelInput:
    """Pure-data description of one Comfy LTX model-call input geometry."""

    x: torch.Tensor                       # [B,C,T_materialized,H,W]
    denoise_mask: torch.Tensor            # [B,1,T_materialized,H,W], holes=-1
    keyframe_idxs: torch.Tensor | None    # [B,3,N_suffix_tokens,2], pixel units (NOT /fps)
    generated_keyframes: dict[str, int] | None
    frame_rate: float
    base_frames: int
    appended_frames: int
    appended_tokens_before_filter: int
    appended_tokens_after_filter: int
    total_tokens_after_filter: int
    operation_frames: tuple[dict[str, Any], ...]
    version: int = BRIDGE_VERSION


def _unpatchify_tokens(tokens: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
    """Inverse of our patch_size=1 video patchification: [B,FHW,C] -> [B,C,F,H,W]."""
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,N,C] tokens, got {tuple(tokens.shape)}.")
    b, n, c = tokens.shape
    expected = int(frames) * int(height) * int(width)
    if n != expected:
        raise ValueError(f"Cannot unpatchify {n} tokens as F={frames}, H={height}, W={width} ({expected} expected).")
    return tokens.reshape(b, frames, height, width, c).permute(0, 4, 1, 2, 3).contiguous()


def _unpatchify_mask(tokens: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[-1] != 1:
        raise ValueError(f"Expected [B,N,1] mask tokens, got {tuple(tokens.shape)}.")
    b, n, _ = tokens.shape
    expected = int(frames) * int(height) * int(width)
    if n != expected:
        raise ValueError(f"Cannot unpatchify {n} mask tokens as F={frames}, H={height}, W={width} ({expected} expected).")
    return tokens.reshape(b, frames, height, width, 1).permute(0, 4, 1, 2, 3).contiguous()


def _unpatchify_positions(tokens: torch.Tensor, frames: int, height: int, width: int) -> torch.Tensor:
    """[B,3,FHW,2] -> [B,3,F,H,W,2]."""
    if tokens.ndim != 4 or tokens.shape[1] != 3 or tokens.shape[-1] != 2:
        raise ValueError(f"Expected positions [B,3,N,2], got {tuple(tokens.shape)}.")
    b, _, n, _ = tokens.shape
    expected = int(frames) * int(height) * int(width)
    if n != expected:
        raise ValueError(f"Cannot unpatchify {n} position tokens as F={frames}, H={height}, W={width} ({expected} expected).")
    return tokens.reshape(b, 3, frames, height, width, 2).contiguous()


def _flatten_video(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], -1, x.shape[1])


def _flatten_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask.permute(0, 2, 3, 4, 1).reshape(mask.shape[0], -1, 1)


def _base_pixel_positions(
    *,
    batch: int,
    frames: int,
    height: int,
    width: int,
    scale_factors: tuple[int, int, int],
    causal_fix: bool,
    device: torch.device,
) -> torch.Tensor:
    """Emulate Comfy latent_to_pixel_coords for patch_size=1, start/end coordinates."""
    t = torch.arange(frames, device=device)
    y = torch.arange(height, device=device)
    x = torch.arange(width, device=device)
    tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
    starts = torch.stack([tt, yy, xx], dim=0)
    ends = starts + 1
    coords = torch.stack([starts, ends], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1)
    coords = coords.reshape(batch, 3, frames * height * width, 2)
    scale = torch.tensor(scale_factors, device=device, dtype=coords.dtype).view(1, 3, 1, 1)
    pixel = coords * scale
    if causal_fix:
        pixel[:, 0, ...] = (pixel[:, 0, ...] + 1 - scale_factors[0]).clamp(min=0)
    return pixel.to(torch.float32)


def _require_noised_state(state: dict[str, Any]) -> dict[str, Any]:
    token_state = state.get("token_state")
    if token_state is None:
        raise ValueError("Official state has no token_state.")
    if token_state.get(NOISER_METADATA_KEY) is None:
        raise ValueError(
            "Model-call materialization requires the Gaussian-noised official state. "
            "Apply 'LTX Official: Gaussian Noiser (DFR)' first."
        )
    return token_state


def _operation_appended_frames(op: dict[str, Any], base_h: int, base_w: int) -> tuple[int, int, int]:
    """Return (frames, source_h, source_w) for an appended operation."""
    kind = op.get("type")
    if kind == "VideoConditionByKeyframeIndex":
        shape = tuple(op["conditioning_shape"])
        return int(shape[2]), int(shape[3]), int(shape[4])
    if kind == "VideoGeneratedKeyframeSlots":
        return int(op["num_keyframes"]), int(base_h), int(base_w)
    if kind == "VideoConditionByReferenceLatent":
        shape = tuple(op["reference_shape"])
        return int(shape[2]), int(shape[3]), int(shape[4])
    raise ValueError(f"Unsupported appended operation for Comfy model bridge: {kind!r}.")


def materialize_comfy_model_input(noised_official_state: dict[str, Any]) -> DFRComfyModelInput:
    """Translate the official token state into Comfy LTX's 5-D+suffixed-position representation."""
    state = get_official_state(noised_official_state)
    token_state = _require_noised_state(state)

    base_b, base_c, base_t, base_h, base_w = [int(v) for v in state["base_shape"]]
    base_count = int(token_state["base_token_count"])
    expected_base = base_t * base_h * base_w
    if base_count != expected_base:
        raise ValueError(f"base_token_count={base_count}, expected {expected_base} from base_shape={state['base_shape']}.")

    latent_tokens = token_state["latent"]
    mask_tokens = token_state["denoise_mask"]
    pos_tokens = token_state["positions"]
    if latent_tokens.shape[0] != base_b or latent_tokens.shape[-1] != base_c:
        raise ValueError("Official token-state batch/channel geometry does not match base shape.")

    # Materialize the untouched/base video first.
    base_x = _unpatchify_tokens(latent_tokens[:, :base_count], base_t, base_h, base_w)
    base_mask = _unpatchify_mask(mask_tokens[:, :base_count], base_t, base_h, base_w)
    x_groups = [base_x]
    mask_groups = [base_mask]
    keyframe_pos_groups: list[torch.Tensor] = []
    op_frames: list[dict[str, Any]] = []

    generated_meta = None
    generated_layout = token_state.get("generated_keyframe_layout")

    # Find only operations that appended token groups, preserving exact official order.
    appended_ops = [
        op for op in state.get("operations", [])
        if op.get("type") in {
            "VideoConditionByKeyframeIndex",
            "VideoGeneratedKeyframeSlots",
            "VideoConditionByReferenceLatent",
        }
    ]

    cursor_token = base_count
    cursor_frame = base_t
    fps = float(token_state.get("fps") or state.get("fps") or 24.0)
    appended_before_filter = 0
    appended_after_filter = 0

    for op in appended_ops:
        start = int(op["appended_token_start"])
        count = int(op["appended_token_count"])
        if start != cursor_token:
            raise ValueError(
                f"Official appended token groups are not contiguous/in-order: expected start {cursor_token}, "
                f"operation {op.get('type')} starts at {start}."
            )
        stop = start + count
        if stop > latent_tokens.shape[1]:
            raise ValueError(f"Operation {op.get('type')} token slice [{start}:{stop}] exceeds token-state length.")

        frames, src_h, src_w = _operation_appended_frames(op, base_h, base_w)
        expected_count = frames * src_h * src_w
        if count != expected_count:
            raise ValueError(
                f"Operation {op.get('type')} records {count} tokens but shape implies {expected_count}."
            )

        group_latent_tokens = latent_tokens[:, start:stop]
        group_mask_tokens = mask_tokens[:, start:stop]
        group_pos_tokens = pos_tokens[:, :, start:stop]
        group_x_src = _unpatchify_tokens(group_latent_tokens, frames, src_h, src_w)
        group_mask_src = _unpatchify_mask(group_mask_tokens, frames, src_h, src_w)
        group_pos_src = _unpatchify_positions(group_pos_tokens, frames, src_h, src_w)

        kind = op["type"]
        sparse = src_h != base_h or src_w != base_w
        if sparse:
            if kind != "VideoConditionByReferenceLatent":
                raise ValueError(
                    f"Only VideoConditionByReferenceLatent may use a smaller spatial grid; {kind} has {src_h}x{src_w}."
                )
            factor = int(op.get("downscale_factor", 1))
            if src_h * factor != base_h or src_w * factor != base_w:
                raise ValueError(
                    f"Reference grid {src_h}x{src_w} with downscale_factor={factor} does not map to target {base_h}x{base_w}."
                )
            group_x = torch.zeros(
                (base_b, base_c, frames, base_h, base_w),
                device=group_x_src.device,
                dtype=group_x_src.dtype,
            )
            group_mask = torch.full(
                (base_b, 1, frames, base_h, base_w),
                -1.0,
                device=group_mask_src.device,
                dtype=group_mask_src.dtype,
            )
            group_pos = torch.zeros(
                (base_b, 3, frames, base_h, base_w, 2),
                device=group_pos_src.device,
                dtype=torch.float32,
            )
            group_x[..., ::factor, ::factor] = group_x_src
            group_mask[..., ::factor, ::factor] = group_mask_src
            group_pos[:, :, :, ::factor, ::factor, :] = group_pos_src
            surviving = frames * src_h * src_w
        else:
            group_x = group_x_src
            group_mask = group_mask_src
            group_pos = group_pos_src
            surviving = count

        # Comfy's keyframe_idxs uses pixel-time coordinates; our official state already divided time by fps.
        group_pos = group_pos.clone()
        group_pos[:, 0, ...] *= fps
        keyframe_pos_groups.append(group_pos.reshape(base_b, 3, frames * base_h * base_w, 2))
        x_groups.append(group_x)
        mask_groups.append(group_mask)

        if kind == "VideoGeneratedKeyframeSlots":
            if generated_meta is not None:
                raise ValueError("Multiple generated-keyframe groups are not supported by the official DFR parity path.")
            generated_meta = {
                "first_latent_frame": int(cursor_frame),
                "num_keyframes": int(frames),
                "tokens_per_frame": int(base_h * base_w),
            }
            if generated_layout is None:
                raise ValueError("Generated-keyframe operation exists but token_state.generated_keyframe_layout is missing.")
            if int(generated_layout["first_token"]) != start:
                raise ValueError("generated_keyframe_layout first_token does not match the operation token start.")

        op_frames.append(
            {
                "type": kind,
                "first_materialized_frame": int(cursor_frame),
                "materialized_frames": int(frames),
                "source_height": int(src_h),
                "source_width": int(src_w),
                "sparse": bool(sparse),
                "surviving_tokens": int(surviving),
                "pre_filter_tokens": int(frames * base_h * base_w),
            }
        )
        appended_before_filter += frames * base_h * base_w
        appended_after_filter += surviving
        cursor_token = stop
        cursor_frame += frames

    if cursor_token != latent_tokens.shape[1]:
        raise ValueError(
            f"Model bridge consumed {cursor_token} official tokens but token-state contains {latent_tokens.shape[1]}. "
            "An appended conditioning type is missing from the adapter."
        )

    x = torch.cat(x_groups, dim=2)
    denoise_mask = torch.cat(mask_groups, dim=2).to(torch.float32)
    keyframe_idxs = torch.cat(keyframe_pos_groups, dim=2) if keyframe_pos_groups else None

    if keyframe_idxs is not None and keyframe_idxs.shape[2] != appended_before_filter:
        raise ValueError(
            f"keyframe_idxs contains {keyframe_idxs.shape[2]} tokens, expected {appended_before_filter} pre-filter suffix tokens."
        )

    return DFRComfyModelInput(
        x=x,
        denoise_mask=denoise_mask,
        keyframe_idxs=keyframe_idxs,
        generated_keyframes=generated_meta,
        frame_rate=official_conditioning_fps(fps),
        base_frames=base_t,
        appended_frames=cursor_frame - base_t,
        appended_tokens_before_filter=appended_before_filter,
        appended_tokens_after_filter=appended_after_filter,
        total_tokens_after_filter=base_count + appended_after_filter,
        operation_frames=tuple(op_frames),
    )


def emulate_comfy_process_input(model_input: DFRComfyModelInput, *, sigma: float) -> dict[str, torch.Tensor | None]:
    """Emulate the geometry/timestep part of current Comfy LTXVModel._process_input.

    This intentionally stops before patchify_proj / transformer math.
    """
    x = model_input.x
    mask_5d = model_input.denoise_mask
    b, c, t, h, w = x.shape
    tokens = _flatten_video(x)
    mask_tokens = _flatten_mask(mask_5d)
    pixel_coords = _base_pixel_positions(
        batch=b,
        frames=t,
        height=h,
        width=w,
        scale_factors=(8, 32, 32),
        causal_fix=True,
        device=x.device,
    )

    grid_mask = None
    if model_input.keyframe_idxs is not None and model_input.keyframe_idxs.shape[2] > 0:
        tokens_per_frame = h * w
        if model_input.keyframe_idxs.shape[2] % tokens_per_frame != 0:
            raise ValueError("keyframe_idxs is not a whole number of materialized latent frames.")
        grid_mask = ~torch.any(mask_tokens < 0, dim=-1)[0]
        tokens = tokens[:, grid_mask, :]
        pixel_coords = pixel_coords[:, :, grid_mask, ...]
        suffix_grid_mask = grid_mask[-model_input.keyframe_idxs.shape[2]:]
        kf = model_input.keyframe_idxs[..., suffix_grid_mask, :]
        if kf.shape[2] > 0:
            pixel_coords[:, :, -kf.shape[2]:, :] = kf
        mask_tokens = mask_tokens[:, grid_mask, :]

    # Comfy divides temporal pixel coordinates by frame_rate when preparing RoPE.
    official_style_positions = pixel_coords.to(torch.float32)
    official_style_positions[:, 0, ...] /= float(model_input.frame_rate)

    # Comfy BaseModel LTXV.process_timestep: patchify(mask * scalar timestep).
    timestep_tokens = mask_tokens[..., 0] * float(sigma)

    # Reproduce Comfy's single-frame marker mask semantics before projection.
    temporal_start = pixel_coords[:, 0, :, 0]
    marker_mask = temporal_start == 0
    num_suffix_surviving = 0
    if model_input.keyframe_idxs is not None:
        if grid_mask is None:
            num_suffix_surviving = model_input.keyframe_idxs.shape[2]
        else:
            num_suffix_surviving = int(grid_mask[-model_input.keyframe_idxs.shape[2]:].sum().item())
        if num_suffix_surviving > 0:
            marker_mask[:, -num_suffix_surviving:] = False
    gk = model_input.generated_keyframes
    if gk is not None:
        tpf = h * w
        first_token = int(gk["first_latent_frame"]) * tpf
        num_slot_tokens = int(gk["num_keyframes"]) * tpf
        slots = torch.zeros(t * tpf, dtype=torch.bool, device=x.device)
        slots[first_token:first_token + num_slot_tokens] = True
        if grid_mask is not None:
            slots = slots[grid_mask]
        marker_mask = marker_mask | slots.unsqueeze(0)

    return {
        "tokens": tokens,
        "denoise_mask": mask_tokens,
        "positions": official_style_positions,
        "timesteps": timestep_tokens,
        "marker_mask": marker_mask,
        "grid_mask": grid_mask,
    }


def validate_model_input_roundtrip(noised_official_state: dict[str, Any], model_input: DFRComfyModelInput, *, sigma: float) -> dict[str, Any]:
    """Round-trip the Comfy representation and compare it to the official token state."""
    state = get_official_state(noised_official_state)
    token_state = _require_noised_state(state)
    emu = emulate_comfy_process_input(model_input, sigma=sigma)

    expected_latent = token_state["latent"].to(device=emu["tokens"].device, dtype=emu["tokens"].dtype)
    expected_mask = token_state["denoise_mask"].to(device=emu["denoise_mask"].device, dtype=emu["denoise_mask"].dtype)
    expected_positions = token_state["positions"].to(device=emu["positions"].device, dtype=emu["positions"].dtype)
    state_fps = float(token_state.get("fps") or state.get("fps") or 24.0)
    expected_positions = model_video_positions(expected_positions, state_fps)
    expected_timesteps = expected_mask[..., 0] * float(sigma)

    if emu["tokens"].shape != expected_latent.shape:
        raise ValueError(f"Round-trip token shape {tuple(emu['tokens'].shape)} != official {tuple(expected_latent.shape)}.")
    if emu["denoise_mask"].shape != expected_mask.shape:
        raise ValueError(f"Round-trip mask shape {tuple(emu['denoise_mask'].shape)} != official {tuple(expected_mask.shape)}.")
    if emu["positions"].shape != expected_positions.shape:
        raise ValueError(f"Round-trip positions shape {tuple(emu['positions'].shape)} != official {tuple(expected_positions.shape)}.")

    latent_err = float((emu["tokens"] - expected_latent).abs().max().item()) if expected_latent.numel() else 0.0
    mask_err = float((emu["denoise_mask"] - expected_mask).abs().max().item()) if expected_mask.numel() else 0.0
    pos_err = float((emu["positions"] - expected_positions).abs().max().item()) if expected_positions.numel() else 0.0
    timestep_err = float((emu["timesteps"] - expected_timesteps).abs().max().item()) if expected_timesteps.numel() else 0.0

    marker_err = 0.0
    expected_kf = token_state.get("keyframes_mask")
    if expected_kf is not None:
        expected_marker = expected_kf[..., 0].to(device=emu["marker_mask"].device) > 0.5
        if emu["marker_mask"].shape != expected_marker.shape:
            raise ValueError(
                f"Comfy marker-mask shape {tuple(emu['marker_mask'].shape)} != official keyframes_mask {tuple(expected_marker.shape)}."
            )
        marker_err = float((emu["marker_mask"] != expected_marker).float().max().item()) if expected_marker.numel() else 0.0

    generated_ok = True
    layout = token_state.get("generated_keyframe_layout")
    if layout is None:
        generated_ok = model_input.generated_keyframes is None
    else:
        gk = model_input.generated_keyframes
        generated_ok = (
            gk is not None
            and int(gk["num_keyframes"]) == int(layout["num_keyframes"])
            and int(gk["tokens_per_frame"]) == int(layout["tokens_per_keyframe"])
        )

    return {
        "passed": latent_err == 0.0 and mask_err == 0.0 and pos_err == 0.0 and timestep_err == 0.0 and marker_err == 0.0 and generated_ok,
        "latent_error": latent_err,
        "mask_error": mask_err,
        "position_error": pos_err,
        "timestep_error": timestep_err,
        "marker_error": marker_err,
        "generated_metadata_ok": bool(generated_ok),
        "materialized_shape": tuple(int(v) for v in model_input.x.shape),
        "official_token_shape": tuple(int(v) for v in expected_latent.shape),
        "suffix_pre_filter": int(model_input.appended_tokens_before_filter),
        "suffix_surviving": int(model_input.appended_tokens_after_filter),
        "grid_filtered_tokens": int((~emu["grid_mask"]).sum().item()) if emu["grid_mask"] is not None else 0,
    }
