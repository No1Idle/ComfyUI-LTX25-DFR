"""Capture reusable encoded user keyframes for Temporal DFR.

The official pipeline keeps the original image-conditioning inputs and re-encodes
them for every temporal tile.  Comfy's Spatial DFR path has already encoded those
images at the source stage's spatial resolution.  Temporal upscaling does not
change H/W, so this module extracts those encoded still latents once and carries
their original pixel-frame indices for exact round/tile rebasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

if "." in (__package__ or ""):  # Normal Comfy custom-node package import.
    from ..DFR_Spatial.latent_state import OFFICIAL_STATE_KEY, get_official_state
else:  # Standalone temporal regression tests.
    from DFR_Spatial.latent_state import OFFICIAL_STATE_KEY, get_official_state


@dataclass(frozen=True)
class DFRTemporalUserKeyframe:
    pixel_frame_index: int
    strength: float
    latent: torch.Tensor


_NON_USER_OPERATIONS = {
    "VideoGeneratedKeyframeSlots",
    "VideoConditionByReferenceLatent",
}


def _unpatchify_conditioning_tokens(
    tokens: torch.Tensor,
    shape: tuple[int, int, int, int, int],
) -> torch.Tensor:
    batch, channels, frames, height, width = shape
    expected = (batch, frames * height * width, channels)
    if tuple(tokens.shape) != expected:
        raise ValueError(
            f"User keyframe token slice has shape {tuple(tokens.shape)}, expected {expected}."
        )
    return (
        tokens.reshape(batch, frames, height, width, channels)
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )


def validate_temporal_user_keyframes(
    values: tuple[DFRTemporalUserKeyframe, ...],
    *,
    source_video: torch.Tensor,
) -> tuple[DFRTemporalUserKeyframe, ...]:
    if not torch.is_tensor(source_video) or source_video.ndim != 5:
        raise ValueError("source_video must be a [B,C,T,H,W] tensor.")
    for index, value in enumerate(values):
        if not isinstance(value, DFRTemporalUserKeyframe):
            raise ValueError(
                f"user_keyframes[{index}] has unexpected type {type(value).__name__}."
            )
        if int(value.pixel_frame_index) < 0:
            raise ValueError("User keyframe pixel indices must be non-negative.")
        if not 0.0 <= float(value.strength) <= 1.0:
            raise ValueError(f"User keyframe strength must be in [0,1], got {value.strength}.")
        expected_shape = (
            int(source_video.shape[0]),
            int(source_video.shape[1]),
            1,
            int(source_video.shape[3]),
            int(source_video.shape[4]),
        )
        if not torch.is_tensor(value.latent) or tuple(value.latent.shape) != expected_shape:
            raise ValueError(
                f"user_keyframes[{index}] latent has shape {getattr(value.latent, 'shape', None)}, "
                f"expected {expected_shape}."
            )
    return values


def capture_temporal_user_keyframes(
    user_conditioning_state: dict[str, Any] | None,
    *,
    source_video: torch.Tensor,
) -> tuple[DFRTemporalUserKeyframe, ...]:
    """Extract ordinary encoded image guides from a Spatial official state.

    The input may be the Stage-1 state after generated slots or the Stage-2
    conditioning state after slots/reference; those DFR-only operations are
    ignored. Only the ordinary frame-0 and later-keyframe operations are kept.
    """
    if user_conditioning_state is None:
        return ()
    if not isinstance(user_conditioning_state, dict):
        raise ValueError(
            "user_conditioning_state must be a Spatial DFR LATENT carrying official conditioning metadata."
        )
    samples = user_conditioning_state.get("samples")
    if not torch.is_tensor(samples) or samples.ndim != 5:
        raise ValueError("user_conditioning_state must contain [B,C,T,H,W] tensor 'samples'.")
    if tuple(samples.shape) != tuple(source_video.shape):
        raise ValueError(
            "User conditioning state must use the same source-stage geometry as the temporal handoff: "
            f"state={tuple(samples.shape)}, source={tuple(source_video.shape)}."
        )

    try:
        state = get_official_state(user_conditioning_state)
    except ValueError as exc:
        if OFFICIAL_STATE_KEY not in user_conditioning_state:
            raise ValueError(
                "user_conditioning_state has no Spatial DFR official conditioning metadata."
            ) from exc
        raise
    token_state = state.get("token_state")
    if not isinstance(token_state, dict):
        raise ValueError("user_conditioning_state has no materialized video token state.")

    captured: list[DFRTemporalUserKeyframe] = []
    for operation in state.get("operations", []):
        op_type = operation.get("type")
        if op_type in _NON_USER_OPERATIONS:
            continue
        if op_type == "VideoConditionByLatentIndex":
            latent_idx = int(operation.get("latent_idx", -1))
            latent_frames = int(operation.get("latent_frames", -1))
            if latent_idx != 0 or latent_frames != 1:
                raise ValueError(
                    "Temporal DFR currently supports the official still-image frame-0 conditioning "
                    f"contract, got latent_idx={latent_idx}, latent_frames={latent_frames}."
                )
            latent = state["clean_latent"][:, :, :1].detach().clone()
            captured.append(
                DFRTemporalUserKeyframe(
                    pixel_frame_index=0,
                    strength=float(operation["strength"]),
                    latent=latent,
                )
            )
            continue
        if op_type == "VideoConditionByKeyframeIndex":
            if int(operation.get("num_pixel_frames", 1)) != 1:
                raise ValueError("Temporal DFR user-keyframe rebasing currently supports still images only.")
            shape = tuple(int(x) for x in operation.get("conditioning_shape", ()))
            if len(shape) != 5 or shape[2] != 1:
                raise ValueError(f"Malformed user-keyframe conditioning_shape={shape}.")
            first = int(operation.get("appended_token_start", -1))
            count = int(operation.get("appended_token_count", -1))
            if first < 0 or count < 1:
                raise ValueError("User-keyframe operation lacks its recorded token slice.")
            tokens = token_state["clean_latent"][:, first : first + count]
            latent = _unpatchify_conditioning_tokens(tokens, shape).detach().clone()
            captured.append(
                DFRTemporalUserKeyframe(
                    pixel_frame_index=int(operation["frame_idx"]),
                    strength=float(operation["strength"]),
                    latent=latent,
                )
            )
            continue
        raise ValueError(
            "user_conditioning_state contains an unsupported operation for Temporal DFR: "
            f"{op_type!r}."
        )

    return validate_temporal_user_keyframes(tuple(captured), source_video=source_video)
