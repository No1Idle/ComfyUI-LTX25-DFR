"""Strict Lightricks-style state helpers for the Spatial DFR port.

This module keeps the official LTX conditioning state separate from ComfyUI's
normal ``LATENT['samples']`` field.  Phase A validates the official conditioning
semantics first; materializing that state for sampling is a later step.

The state stores two views:
  * base latent-space tensors in normal LTX layout ``[B,C,T,H,W]``;
  * an optional patchified token-state view mirroring the official LTX
    ``LatentState`` after ``VideoLatentTools.create_initial_state()``.

Reference semantics:
  Lightricks/LTX-2
  - packages/ltx-core/src/ltx_core/conditioning/types/latent_cond.py
  - packages/ltx-core/src/ltx_core/conditioning/types/keyframe_cond.py
  - packages/ltx-core/src/ltx_core/tools.py
  - packages/ltx-core/src/ltx_core/components/patchifiers.py
"""

from __future__ import annotations

from typing import Any

import torch


OFFICIAL_STATE_KEY = "_ltx_dfr_official_state"
OFFICIAL_STATE_VERSION = 2
DEFAULT_SCALE_FACTORS = (8, 32, 32)  # time, height, width


def _validate_video_tensor(tensor: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(tensor) or tensor.ndim != 5:
        raise ValueError(
            f"{name} must be a normal LTX video latent tensor shaped [B,C,T,H,W]; "
            f"got {getattr(tensor, 'shape', None)}."
        )


def _clone_tensor_or_none(value: torch.Tensor | None) -> torch.Tensor | None:
    return value.clone() if torch.is_tensor(value) else None


def _patchify_video(latents: torch.Tensor) -> torch.Tensor:
    _validate_video_tensor(latents, "tensor to patchify")
    b, c, t, h, w = latents.shape
    return latents.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)


def _patchify_mask(mask: torch.Tensor) -> torch.Tensor:
    _validate_video_tensor(mask, "mask to patchify")
    if mask.shape[1] != 1:
        raise ValueError(f"Expected a single-channel mask shaped [B,1,T,H,W], got {tuple(mask.shape)}.")
    b, _, t, h, w = mask.shape
    return mask.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, 1)


def _get_patch_grid_bounds(shape: tuple[int, int, int, int, int], device: torch.device) -> torch.Tensor:
    """Return latent-space [start,end) bounds shaped [B,3,N,2]."""
    batch, _, frames, height, width = shape
    grid_coords = torch.meshgrid(
        torch.arange(start=0, end=frames, step=1, device=device),
        torch.arange(start=0, end=height, step=1, device=device),
        torch.arange(start=0, end=width, step=1, device=device),
        indexing="ij",
    )
    patch_starts = torch.stack(grid_coords, dim=0)  # [3,F,H,W]
    patch_size_delta = torch.tensor([1, 1, 1], device=device, dtype=patch_starts.dtype).view(3, 1, 1, 1)
    patch_ends = patch_starts + patch_size_delta
    latent_coords = torch.stack((patch_starts, patch_ends), dim=-1)  # [3,F,H,W,2]
    latent_coords = latent_coords.unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1).reshape(batch, 3, frames * height * width, 2)
    return latent_coords


def _get_pixel_coords(
    latent_coords: torch.Tensor,
    scale_factors: tuple[int, int, int],
    causal_fix: bool = False,
) -> torch.Tensor:
    scale_tensor = torch.tensor(scale_factors, device=latent_coords.device, dtype=latent_coords.dtype).view(1, 3, 1, 1)
    pixel_coords = latent_coords * scale_tensor
    if causal_fix:
        pixel_coords[:, 0, ...] = (pixel_coords[:, 0, ...] + 1 - scale_factors[0]).clamp(min=0)
    return pixel_coords


def _build_base_positions(
    shape: tuple[int, int, int, int, int],
    device: torch.device,
    fps: float,
    scale_factors: tuple[int, int, int],
    causal_fix: bool,
) -> torch.Tensor:
    latent_coords = _get_patch_grid_bounds(shape, device=device)
    positions = _get_pixel_coords(latent_coords=latent_coords, scale_factors=scale_factors, causal_fix=causal_fix).to(dtype=torch.float32)
    positions[:, 0, ...] = positions[:, 0, ...] / float(fps)
    return positions


def _build_initial_token_state(
    *,
    samples: torch.Tensor,
    clean_latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    fps: float,
    scale_factors: tuple[int, int, int],
    causal_fix: bool,
) -> dict[str, Any]:
    latent_tokens = _patchify_video(samples)
    clean_tokens = _patchify_video(clean_latent)
    mask_tokens = _patchify_mask(denoise_mask).to(dtype=torch.float32)
    positions = _build_base_positions(
        shape=tuple(samples.shape),
        device=samples.device,
        fps=fps,
        scale_factors=scale_factors,
        causal_fix=causal_fix,
    )
    tokens_per_latent_frame = samples.shape[3] * samples.shape[4]
    keyframes_mask = torch.zeros_like(mask_tokens)
    keyframes_mask[:, :tokens_per_latent_frame] = 1.0
    return {
        "latent": latent_tokens,
        "clean_latent": clean_tokens,
        "denoise_mask": mask_tokens,
        "positions": positions,
        "attention_mask": None,
        "keyframes_mask": keyframes_mask,
        "generated_keyframe_layout": None,
        "generated_keyframes": None,
        "frozen": None,
        "base_token_count": int(latent_tokens.shape[1]),
        "tokens_per_latent_frame": int(tokens_per_latent_frame),
        "causal_fix": bool(causal_fix),
        "fps": float(fps),
        "scale_factors": tuple(int(x) for x in scale_factors),
    }


def _clone_token_state(token_state: dict[str, Any] | None) -> dict[str, Any] | None:
    if token_state is None:
        return None
    return {
        "latent": token_state["latent"].clone(),
        "clean_latent": token_state["clean_latent"].clone(),
        "denoise_mask": token_state["denoise_mask"].clone(),
        "positions": token_state["positions"].clone(),
        "attention_mask": _clone_tensor_or_none(token_state.get("attention_mask")),
        "keyframes_mask": _clone_tensor_or_none(token_state.get("keyframes_mask")),
        "generated_keyframe_layout": token_state.get("generated_keyframe_layout"),
        "generated_keyframes": _clone_tensor_or_none(token_state.get("generated_keyframes")),
        "frozen": token_state.get("frozen"),
        "base_token_count": int(token_state.get("base_token_count", 0)),
        "tokens_per_latent_frame": int(token_state.get("tokens_per_latent_frame", 0)),
        "causal_fix": bool(token_state.get("causal_fix", True)),
        "fps": float(token_state.get("fps", 24.0)),
        "scale_factors": tuple(int(x) for x in token_state.get("scale_factors", DEFAULT_SCALE_FACTORS)),
        "gaussian_noiser": dict(token_state["gaussian_noiser"]) if isinstance(token_state.get("gaussian_noiser"), dict) else None,
    }


def _new_official_state(latent: dict[str, Any]) -> dict[str, Any]:
    if "samples" not in latent:
        raise ValueError("Input LATENT has no 'samples' tensor.")

    samples = latent["samples"]
    _validate_video_tensor(samples, "target latent")

    clean_latent = samples.clone()
    denoise_mask = torch.ones(
        (samples.shape[0], 1, samples.shape[2], samples.shape[3], samples.shape[4]),
        dtype=torch.float32,
        device=samples.device,
    )

    return {
        "version": OFFICIAL_STATE_VERSION,
        "base_shape": tuple(samples.shape),
        "clean_latent": clean_latent,
        "denoise_mask": denoise_mask,
        "operations": [],
        "token_state": None,
        "fps": None,
        "scale_factors": DEFAULT_SCALE_FACTORS,
        "causal_fix": True,
    }


def clone_or_create_official_state(latent: dict[str, Any]) -> dict[str, Any]:
    """Return a cloned strict Phase-A state carried inside a Comfy LATENT dict."""
    if "samples" not in latent:
        raise ValueError("Input LATENT has no 'samples' tensor.")

    samples = latent["samples"]
    _validate_video_tensor(samples, "target latent")

    existing = latent.get(OFFICIAL_STATE_KEY)
    if existing is None:
        if "noise_mask" in latent:
            raise ValueError(
                "Strict LTX DFR Phase-A input already contains a stock ComfyUI 'noise_mask'. "
                "Start this official-conditioning branch from the unconditioned LTX latent, "
                "not from LTXVAddGuide / LTXVImgToVideoInplace / the old Blend-Replace nodes."
            )
        return _new_official_state(latent)

    if not isinstance(existing, dict):
        raise ValueError(f"Malformed {OFFICIAL_STATE_KEY}: expected dict.")
    if existing.get("version") != OFFICIAL_STATE_VERSION:
        raise ValueError(
            f"Unsupported official-state version {existing.get('version')}; expected {OFFICIAL_STATE_VERSION}."
        )
    if tuple(existing.get("base_shape", ())) != tuple(samples.shape):
        raise ValueError(
            "Official LTX state was created for a different target latent shape: "
            f"state={existing.get('base_shape')}, current={tuple(samples.shape)}."
        )

    clean_latent = existing.get("clean_latent")
    denoise_mask = existing.get("denoise_mask")
    _validate_video_tensor(clean_latent, "official clean_latent")
    _validate_video_tensor(denoise_mask, "official denoise_mask")

    if clean_latent.shape != samples.shape:
        raise ValueError(
            f"official clean_latent shape {tuple(clean_latent.shape)} does not match target {tuple(samples.shape)}."
        )
    expected_mask_shape = (samples.shape[0], 1, samples.shape[2], samples.shape[3], samples.shape[4])
    if tuple(denoise_mask.shape) != expected_mask_shape:
        raise ValueError(
            f"official denoise_mask shape {tuple(denoise_mask.shape)} does not match expected {expected_mask_shape}."
        )

    return {
        "version": OFFICIAL_STATE_VERSION,
        "base_shape": tuple(samples.shape),
        "clean_latent": clean_latent.clone(),
        "denoise_mask": denoise_mask.clone(),
        "operations": list(existing.get("operations", [])),
        "token_state": _clone_token_state(existing.get("token_state")),
        "fps": existing.get("fps"),
        "scale_factors": tuple(int(x) for x in existing.get("scale_factors", DEFAULT_SCALE_FACTORS)),
        "causal_fix": bool(existing.get("causal_fix", True)),
    }


def ensure_token_state(
    state: dict[str, Any],
    *,
    target_samples: torch.Tensor,
    fps: float | None,
    scale_factors: tuple[int, int, int] = DEFAULT_SCALE_FACTORS,
    causal_fix: bool | None = None,
) -> dict[str, Any]:
    """Ensure the state carries the patchified token-state view.

    If the token view does not exist yet, it is created from the current base
    clean_latent/denoise_mask tensors.  If it already exists, fps/scale factors
    must stay consistent.
    """
    token_state = state.get("token_state")
    if token_state is None:
        resolved_fps = float(fps if fps is not None else state.get("fps") or 24.0)
        resolved_scale = tuple(int(x) for x in (scale_factors or state.get("scale_factors") or DEFAULT_SCALE_FACTORS))
        resolved_causal_fix = bool(state.get("causal_fix") if causal_fix is None else causal_fix)
        token_state = _build_initial_token_state(
            samples=target_samples,
            clean_latent=state["clean_latent"],
            denoise_mask=state["denoise_mask"],
            fps=resolved_fps,
            scale_factors=resolved_scale,
            causal_fix=resolved_causal_fix,
        )
        state["token_state"] = token_state
        state["fps"] = resolved_fps
        state["scale_factors"] = resolved_scale
        state["causal_fix"] = resolved_causal_fix
        return token_state

    if fps is not None:
        resolved_fps = float(fps)
        if abs(float(token_state.get("fps", resolved_fps)) - resolved_fps) > 1e-6:
            raise ValueError(
                f"Official token-state already uses fps={token_state.get('fps')}, got incompatible fps={resolved_fps}."
            )
    resolved_scale = tuple(int(x) for x in (scale_factors or token_state.get("scale_factors") or DEFAULT_SCALE_FACTORS))
    if tuple(token_state.get("scale_factors", DEFAULT_SCALE_FACTORS)) != resolved_scale:
        raise ValueError(
            f"Official token-state already uses scale_factors={token_state.get('scale_factors')}, "
            f"got incompatible scale_factors={resolved_scale}."
        )
    return token_state


def _update_base_token_slices(state: dict[str, Any], target_samples: torch.Tensor) -> None:
    """If a token-state already exists, refresh the base target-token slices from the current base tensors."""
    token_state = state.get("token_state")
    if token_state is None:
        return
    base_count = int(token_state["base_token_count"])
    token_state["clean_latent"][:, :base_count] = _patchify_video(state["clean_latent"]).to(
        device=token_state["clean_latent"].device,
        dtype=token_state["clean_latent"].dtype,
    )
    token_state["denoise_mask"][:, :base_count] = _patchify_mask(state["denoise_mask"]).to(
        device=token_state["denoise_mask"].device,
        dtype=token_state["denoise_mask"].dtype,
    )
    token_state["latent"][:, :base_count] = _patchify_video(target_samples).to(
        device=token_state["latent"].device,
        dtype=token_state["latent"].dtype,
    )


def apply_video_condition_by_latent_index(
    target_latent: dict[str, Any],
    conditioning_latent: dict[str, Any],
    strength: float,
    latent_idx: int,
) -> dict[str, Any]:
    """Native-Comfy representation of official VideoConditionByLatentIndex."""
    if "samples" not in conditioning_latent:
        raise ValueError("conditioning_latent has no 'samples' tensor.")

    target_samples = target_latent.get("samples")
    cond_samples = conditioning_latent["samples"]
    _validate_video_tensor(target_samples, "target latent")
    _validate_video_tensor(cond_samples, "conditioning latent")

    strength = float(strength)
    latent_idx = int(latent_idx)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0,1], got {strength}.")
    if latent_idx < 0:
        raise ValueError(
            "Official VideoConditionByLatentIndex uses a non-negative latent_idx. "
            f"Got {latent_idx}."
        )

    cond_batch, cond_channels, cond_frames, cond_height, cond_width = cond_samples.shape
    tgt_batch, tgt_channels, tgt_frames, tgt_height, tgt_width = target_samples.shape

    if (cond_batch, cond_channels, cond_height, cond_width) != (
        tgt_batch,
        tgt_channels,
        tgt_height,
        tgt_width,
    ):
        raise ValueError(
            "Conditioning latent does not match the official VideoConditionByLatentIndex shape contract. "
            "Expected matching batch/channels/height/width. "
            f"conditioning={tuple(cond_samples.shape)}, target={tuple(target_samples.shape)}."
        )

    stop_idx = latent_idx + cond_frames
    if latent_idx >= tgt_frames or stop_idx > tgt_frames:
        raise ValueError(
            f"Conditioning latent spans latent frames [{latent_idx}, {stop_idx}), but target has only {tgt_frames} latent frames."
        )

    state = clone_or_create_official_state(target_latent)
    cond_samples = cond_samples.to(device=state["clean_latent"].device, dtype=state["clean_latent"].dtype)

    state["clean_latent"][:, :, latent_idx:stop_idx, :, :] = cond_samples
    state["denoise_mask"][:, :, latent_idx:stop_idx, :, :] = 1.0 - strength
    _update_base_token_slices(state, target_samples)
    state["operations"].append(
        {
            "type": "VideoConditionByLatentIndex",
            "latent_idx": latent_idx,
            "latent_frames": cond_frames,
            "strength": strength,
        }
    )

    out = target_latent.copy()
    out[OFFICIAL_STATE_KEY] = state
    return out


def apply_video_condition_by_keyframe_index(
    target_latent: dict[str, Any],
    conditioning_latent: dict[str, Any],
    frame_idx: int,
    strength: float,
    fps: float,
    num_pixel_frames: int = 1,
) -> dict[str, Any]:
    """Native-Comfy representation of official VideoConditionByKeyframeIndex."""
    if "samples" not in conditioning_latent:
        raise ValueError("conditioning_latent has no 'samples' tensor.")

    target_samples = target_latent.get("samples")
    keyframes = conditioning_latent["samples"]
    _validate_video_tensor(target_samples, "target latent")
    _validate_video_tensor(keyframes, "conditioning latent")

    frame_idx = int(frame_idx)
    strength = float(strength)
    fps = float(fps)
    num_pixel_frames = int(num_pixel_frames)
    if frame_idx < 0:
        raise ValueError(f"frame_idx must be non-negative, got {frame_idx}.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0,1], got {strength}.")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if num_pixel_frames < 1:
        raise ValueError(f"num_pixel_frames must be >= 1, got {num_pixel_frames}.")
    if keyframes.shape[0] != target_samples.shape[0]:
        raise ValueError(
            f"conditioning_latent batch {keyframes.shape[0]} does not match target batch {target_samples.shape[0]}."
        )
    if keyframes.shape[1] != target_samples.shape[1]:
        raise ValueError(
            f"conditioning_latent channels {keyframes.shape[1]} do not match target channels {target_samples.shape[1]}."
        )

    state = clone_or_create_official_state(target_latent)
    token_state = ensure_token_state(
        state,
        target_samples=target_samples,
        fps=fps,
        scale_factors=tuple(int(x) for x in state.get("scale_factors", DEFAULT_SCALE_FACTORS)),
        causal_fix=bool(state.get("causal_fix", True)),
    )

    tokens = _patchify_video(keyframes).to(device=token_state["clean_latent"].device, dtype=token_state["clean_latent"].dtype)
    latent_coords = _get_patch_grid_bounds(tuple(keyframes.shape), device=keyframes.device)
    positions = _get_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=tuple(int(x) for x in token_state["scale_factors"]),
        causal_fix=bool(token_state["causal_fix"]) if frame_idx == 0 else False,
    )
    positions[:, 0, ...] += frame_idx
    if num_pixel_frames == 1:
        positions[:, 0, ..., 1:] = positions[:, 0, ..., :1] + 1
    positions = positions.to(dtype=torch.float32, device=token_state["positions"].device)
    positions[:, 0, ...] = positions[:, 0, ...] / fps

    denoise_mask = torch.full(
        size=(tokens.shape[0], tokens.shape[1], 1),
        fill_value=1.0 - strength,
        device=tokens.device,
        dtype=tokens.dtype,
    )
    zeros_like_tokens = torch.zeros_like(tokens)

    append_start = int(token_state["latent"].shape[1])
    token_state["latent"] = torch.cat([token_state["latent"], zeros_like_tokens], dim=1)
    token_state["clean_latent"] = torch.cat([token_state["clean_latent"], tokens], dim=1)
    token_state["denoise_mask"] = torch.cat([token_state["denoise_mask"], denoise_mask], dim=1)
    token_state["positions"] = torch.cat([token_state["positions"], positions], dim=2)
    if token_state.get("keyframes_mask") is None:
        token_state["keyframes_mask"] = torch.zeros_like(token_state["denoise_mask"])
    else:
        token_state["keyframes_mask"] = torch.cat(
            [
                token_state["keyframes_mask"],
                torch.zeros(
                    (token_state["keyframes_mask"].shape[0], tokens.shape[1], token_state["keyframes_mask"].shape[2]),
                    device=token_state["keyframes_mask"].device,
                    dtype=token_state["keyframes_mask"].dtype,
                ),
            ],
            dim=1,
        )

    op = {
        "type": "VideoConditionByKeyframeIndex",
        "frame_idx": frame_idx,
        "strength": strength,
        "num_pixel_frames": num_pixel_frames,
        "fps": fps,
        "appended_token_start": append_start,
        "appended_token_count": int(tokens.shape[1]),
        "conditioning_shape": tuple(int(x) for x in keyframes.shape),
    }
    state["operations"].append(op)

    out = target_latent.copy()
    out[OFFICIAL_STATE_KEY] = state
    return out



def _build_single_frame_positions(
    *,
    latent_shape: tuple[int, int, int, int, int],
    pixel_frame_idx: int,
    fps: float,
    scale_factors: tuple[int, int, int],
    causal_fix: bool,
) -> torch.Tensor:
    """Build official positions for a single-frame latent placed at a pixel-frame index."""
    batch, _channels, frames, height, width = latent_shape
    latent_coords = _get_patch_grid_bounds((batch, 1, frames, height, width), device=torch.device('cpu'))
    # move to the correct device by caller afterwards if needed
    positions = _get_pixel_coords(latent_coords=latent_coords, scale_factors=scale_factors, causal_fix=causal_fix)
    positions[:, 0, ...] += int(pixel_frame_idx)
    positions[:, 0, ..., 1:] = positions[:, 0, ..., :1] + 1
    positions = positions.to(dtype=torch.float32)
    positions[:, 0, ...] = positions[:, 0, ...] / float(fps)
    return positions


def apply_video_generated_keyframe_slots(
    target_latent: dict[str, Any],
    pixel_frame_indices: list[int] | tuple[int, ...],
    initial_keyframes: dict[str, Any] | None = None,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Native-Comfy representation of official VideoGeneratedKeyframeSlots.

    Semantics implemented for Phase A:
    - generated slots are **output tokens**, not guide-conditioning tokens;
    - they append extra latent tokens to the token sequence;
    - appended tokens are marked in keyframes_mask because they represent single pixel frames;
    - appended denoise mask is 1.0 (fully generatable, not preserved conditioning);
    - appended ``latent`` tokens are zeros unless ``initial_keyframes`` seeds them;
    - appended ``clean_latent`` tokens are always zeros because denoise_mask=1 makes clean content irrelevant.
    """
    target_samples = target_latent.get("samples")
    _validate_video_tensor(target_samples, "target latent")

    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if not pixel_frame_indices:
        raise ValueError("pixel_frame_indices must contain at least one interior frame index.")

    state = clone_or_create_official_state(target_latent)
    token_state = ensure_token_state(
        state,
        target_samples=target_samples,
        fps=fps,
        scale_factors=tuple(int(x) for x in state.get("scale_factors", DEFAULT_SCALE_FACTORS)),
        causal_fix=bool(state.get("causal_fix", True)),
    )
    if token_state.get("generated_keyframe_layout") is not None:
        raise ValueError(
            "Official state already contains generated_keyframe_layout. "
            "For strict Phase-A parity, add generated keyframe slots only once per state."
        )

    scale_factors = tuple(int(x) for x in token_state.get("scale_factors", DEFAULT_SCALE_FACTORS))
    num_pixel_frames = (target_samples.shape[2] - 1) * scale_factors[0] + 1

    indices = [int(x) for x in pixel_frame_indices]
    if any(x < 0 for x in indices):
        raise ValueError(f"pixel_frame_indices must be non-negative. Got {indices}.")
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError(f"pixel_frame_indices must be strictly increasing. Got {indices}.")
    if indices[-1] >= num_pixel_frames:
        raise ValueError(
            f"Generated keyframe at pixel frame {indices[-1]} is outside the target's "
            f"{num_pixel_frames} frames."
        )

    token_count_per_keyframe = int(token_state["tokens_per_latent_frame"])
    batch = target_samples.shape[0]
    channels = target_samples.shape[1]
    height = target_samples.shape[3]
    width = target_samples.shape[4]
    num_keyframes = len(indices)

    if initial_keyframes is None:
        keyframe_latents = torch.zeros(
            (batch, channels, num_keyframes, height, width),
            device=target_samples.device,
            dtype=target_samples.dtype,
        )
    else:
        init_samples = initial_keyframes.get("samples")
        _validate_video_tensor(init_samples, "initial_keyframes latent")
        expected_shape = (batch, channels, num_keyframes, height, width)
        if tuple(init_samples.shape) != expected_shape:
            raise ValueError(
                f"initial_keyframes must have shape {expected_shape} matching "
                f"(B,C,K,H,W). Got {tuple(init_samples.shape)}."
            )
        keyframe_latents = init_samples.to(device=target_samples.device, dtype=target_samples.dtype)

    appended_tokens = _patchify_video(keyframe_latents)
    appended_mask = torch.ones(
        (batch, num_keyframes * token_count_per_keyframe, 1),
        device=token_state["denoise_mask"].device,
        dtype=token_state["denoise_mask"].dtype,
    )

    position_chunks = []
    for pixel_idx in indices:
        latent_coords = _get_patch_grid_bounds((batch, channels, 1, height, width), device=target_samples.device)
        pos = _get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=scale_factors,
            # Official keyframe slots never apply causal_fix; their temporal span is set explicitly below.
            causal_fix=False,
        )
        pos[:, 0, ...] += int(pixel_idx)
        pos[:, 0, ..., 1:] = pos[:, 0, ..., :1] + 1
        pos = pos.to(dtype=torch.float32, device=token_state["positions"].device)
        pos[:, 0, ...] = pos[:, 0, ...] / fps
        position_chunks.append(pos)
    appended_positions = torch.cat(position_chunks, dim=2)

    append_start = int(token_state["latent"].shape[1])
    token_state["latent"] = torch.cat(
        [token_state["latent"], appended_tokens.to(device=token_state["latent"].device, dtype=token_state["latent"].dtype)],
        dim=1,
    )
    token_state["clean_latent"] = torch.cat(
        [token_state["clean_latent"], torch.zeros_like(appended_tokens).to(
            device=token_state["clean_latent"].device, dtype=token_state["clean_latent"].dtype
        )],
        dim=1,
    )
    token_state["denoise_mask"] = torch.cat([token_state["denoise_mask"], appended_mask], dim=1)
    token_state["positions"] = torch.cat([token_state["positions"], appended_positions], dim=2)
    slot_keyframes_mask = torch.ones(
        (batch, num_keyframes * token_count_per_keyframe, 1),
        device=token_state["denoise_mask"].device,
        dtype=token_state["denoise_mask"].dtype,
    )
    if token_state.get("keyframes_mask") is None:
        token_state["keyframes_mask"] = slot_keyframes_mask
    else:
        token_state["keyframes_mask"] = torch.cat([token_state["keyframes_mask"], slot_keyframes_mask], dim=1)

    layout = {
        "first_token": append_start,
        "num_tokens": num_keyframes * token_count_per_keyframe,
        "num_keyframes": num_keyframes,
        "tokens_per_keyframe": token_count_per_keyframe,
        "pixel_frame_indices": tuple(indices),
    }
    token_state["generated_keyframe_layout"] = layout
    # Official ConditioningItem preserves generated_keyframes here; it is populated only later
    # when generated slot tokens are extracted from the denoised latent state.

    state["operations"].append(
        {
            "type": "VideoGeneratedKeyframeSlots",
            "pixel_frame_indices": tuple(indices),
            "num_keyframes": num_keyframes,
            "has_initial_keyframes": initial_keyframes is not None,
            "fps": fps,
            "appended_token_start": append_start,
            "appended_token_count": num_keyframes * token_count_per_keyframe,
        }
    )

    out = target_latent.copy()
    out[OFFICIAL_STATE_KEY] = state
    return out



def apply_video_condition_by_reference_latent(
    target_latent: dict[str, Any],
    reference_latent: dict[str, Any],
    downscale_factor: int = 1,
    temporal_scale_factor: int = 1,
    strength: float = 1.0,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Native-Comfy representation of official VideoConditionByReferenceLatent.

    Reference: Lightricks/LTX-2
    ``conditioning/types/reference_video_cond.py``.

    The reference is appended as an ordinary clean conditioning group:
      * noisy/current latent receives zero placeholder tokens;
      * clean_latent receives the patchified reference tokens;
      * denoise_mask receives ``1-strength``;
      * reference pixel coordinates are computed with normal LTX causal coordinates,
        then temporal spacing and spatial downscale are translated exactly as upstream;
      * appended reference tokens are NOT marked as keyframes;
      * any existing generated-keyframe layout is preserved unchanged.
    """
    target_samples = target_latent.get("samples")
    ref_samples = reference_latent.get("samples")
    _validate_video_tensor(target_samples, "target latent")
    _validate_video_tensor(ref_samples, "reference latent")

    downscale_factor = int(downscale_factor)
    temporal_scale_factor = int(temporal_scale_factor)
    strength = float(strength)
    fps = float(fps)

    if downscale_factor < 1:
        raise ValueError(f"downscale_factor must be >= 1, got {downscale_factor}.")
    if temporal_scale_factor < 1:
        raise ValueError(f"temporal_scale_factor must be >= 1, got {temporal_scale_factor}.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0,1], got {strength}.")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if ref_samples.shape[0] != target_samples.shape[0]:
        raise ValueError(
            f"reference batch {ref_samples.shape[0]} does not match target batch {target_samples.shape[0]}."
        )
    if ref_samples.shape[1] != target_samples.shape[1]:
        raise ValueError(
            f"reference channels {ref_samples.shape[1]} do not match target channels {target_samples.shape[1]}."
        )

    state = clone_or_create_official_state(target_latent)
    token_state = ensure_token_state(
        state,
        target_samples=target_samples,
        fps=fps,
        scale_factors=tuple(int(x) for x in state.get("scale_factors", DEFAULT_SCALE_FACTORS)),
        causal_fix=bool(state.get("causal_fix", True)),
    )

    tokens = _patchify_video(ref_samples).to(
        device=token_state["clean_latent"].device,
        dtype=token_state["clean_latent"].dtype,
    )

    latent_coords = _get_patch_grid_bounds(tuple(ref_samples.shape), device=ref_samples.device)
    positions = _get_pixel_coords(
        latent_coords=latent_coords,
        scale_factors=tuple(int(x) for x in token_state.get("scale_factors", DEFAULT_SCALE_FACTORS)),
        causal_fix=bool(token_state.get("causal_fix", True)),
    ).to(dtype=torch.float32)

    # Upstream: reference tokens use their own time spacing = target_fps / S.
    positions[:, 0, ...] /= fps / temporal_scale_factor

    # Upstream temporal translation for temporally-subsampled reference streams.
    if temporal_scale_factor != 1:
        # Official code reads latent_state.positions[:,0,0:1,1:2], which is 1/target_fps
        # for the causal first target token. Use the actual token-state value, not a recomputed constant.
        t_target = token_state["positions"][:, 0, 0:1, 1:2].to(
            device=positions.device,
            dtype=torch.float32,
        )
        positions[:, 0, ...] = torch.clamp(
            positions[:, 0, ...] - (temporal_scale_factor - 1) * t_target,
            min=0,
        )

    if downscale_factor != 1:
        positions[:, 1, ...] *= downscale_factor
        positions[:, 2, ...] *= downscale_factor

    positions = positions.to(device=token_state["positions"].device, dtype=torch.float32)
    denoise_mask = torch.full(
        size=(tokens.shape[0], tokens.shape[1], 1),
        fill_value=1.0 - strength,
        device=token_state["denoise_mask"].device,
        dtype=token_state["denoise_mask"].dtype,
    )

    append_start = int(token_state["latent"].shape[1])
    generated_layout_before = token_state.get("generated_keyframe_layout")

    token_state["latent"] = torch.cat(
        [token_state["latent"], torch.zeros_like(tokens, device=token_state["latent"].device, dtype=token_state["latent"].dtype)],
        dim=1,
    )
    token_state["clean_latent"] = torch.cat([token_state["clean_latent"], tokens], dim=1)
    token_state["denoise_mask"] = torch.cat([token_state["denoise_mask"], denoise_mask], dim=1)
    token_state["positions"] = torch.cat([token_state["positions"], positions], dim=2)

    # Official extend_keyframes_mask(..., marked=False): reference tokens are never keyframes.
    if token_state.get("keyframes_mask") is not None:
        token_state["keyframes_mask"] = torch.cat(
            [
                token_state["keyframes_mask"],
                torch.zeros(
                    (token_state["keyframes_mask"].shape[0], tokens.shape[1], token_state["keyframes_mask"].shape[2]),
                    device=token_state["keyframes_mask"].device,
                    dtype=token_state["keyframes_mask"].dtype,
                ),
            ],
            dim=1,
        )

    # DFR uses no ConditioningItemAttentionStrengthWrapper for this reference, so with
    # our current official path attention_mask remains None. If/when explicit attention
    # masking is ported, update_attention_mask semantics will be added as its own layer.
    if token_state.get("attention_mask") is not None:
        raise ValueError(
            "Strict Phase-A VideoConditionByReferenceLatent encountered an existing attention_mask. "
            "Spatial DFR does not use an attention-strength wrapper here; explicit attention-mask "
            "parity has not been enabled in this path yet."
        )

    # Preserve generated slot metadata exactly. Reference tokens can be appended after slots;
    # the recorded slot token slice remains valid because references are appended at the tail.
    token_state["generated_keyframe_layout"] = generated_layout_before

    state["operations"].append(
        {
            "type": "VideoConditionByReferenceLatent",
            "downscale_factor": downscale_factor,
            "temporal_scale_factor": temporal_scale_factor,
            "strength": strength,
            "fps": fps,
            "reference_shape": tuple(int(x) for x in ref_samples.shape),
            "appended_token_start": append_start,
            "appended_token_count": int(tokens.shape[1]),
            "generated_keyframe_layout_preserved": True,
            "generated_keyframe_layout_snapshot": (
                dict(generated_layout_before) if isinstance(generated_layout_before, dict) else generated_layout_before
            ),
        }
    )

    out = target_latent.copy()
    out[OFFICIAL_STATE_KEY] = state
    return out

def extract_generated_keyframes(latent: dict[str, Any]) -> torch.Tensor:
    """Return generated keyframes as an unpatchified [B,C,K,H,W] tensor from the official state."""
    state = get_official_state(latent)
    token_state = state.get("token_state")
    if token_state is None:
        raise ValueError("Official state has no token_state.")
    layout = token_state.get("generated_keyframe_layout")
    if layout is None:
        raise ValueError("Official state has no generated_keyframe_layout.")
    first = int(layout["first_token"])
    num_tokens = int(layout["num_tokens"])
    tpk = int(layout["tokens_per_keyframe"])
    num_keyframes = int(layout["num_keyframes"])
    tokens = token_state["latent"][:, first:first+num_tokens]
    batch = state["base_shape"][0]
    channels = state["base_shape"][1]
    height = state["base_shape"][3]
    width = state["base_shape"][4]
    expected_shape = (batch, num_keyframes * tpk, channels)
    if tuple(tokens.shape) != expected_shape:
        raise ValueError(
            f"Generated keyframe token slice has shape {tuple(tokens.shape)}, expected {expected_shape}."
        )
    return tokens.reshape(batch, num_keyframes, height, width, channels).permute(0, 4, 1, 2, 3).contiguous()

def get_official_state(latent: dict[str, Any]) -> dict[str, Any]:
    state = latent.get(OFFICIAL_STATE_KEY)
    if state is None:
        raise ValueError(
            "Input LATENT has no strict LTX DFR official state. Pass it through an official Phase-A conditioning node first."
        )
    return state
