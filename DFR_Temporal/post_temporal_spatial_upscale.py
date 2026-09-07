"""Latent-only x2 preparation for the post-temporal Spatial DFR epilogue.

The official LTX pipeline spatially upsamples the completed temporal video, but
decodes, Lanczos-resizes and re-encodes its carry keyframes before the final
detailing pass.  This project intentionally supports a lower-cost experimental
variant: run Comfy's native LTX spatial latent upsampler on the video and on all
carried latent keyframes directly.  Optional encoded user guides are spatially
upsampled as well so every later conditioning plane has the target geometry.

This module only prepares the tensors and immutable metadata.  It does not add a
public Comfy node, build the epilogue tile plan, or execute denoising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .temporal_handoff import DFRTemporalHandoff, require_temporal_handoff
from .temporal_user import DFRTemporalUserKeyframe


POST_TEMPORAL_SPATIAL_UPSCALED_VERSION = 1
POST_TEMPORAL_SPATIAL_SCALE = 2
OFFICIAL_EPILOGUE_KEYFRAME_STRENGTH = 1.0


@dataclass(frozen=True)
class DFRPostTemporalSpatialUpscaledHandoff:
    """Prepared target-resolution tensors for the final spatial detailing pass.

    ``source_handoff.video_latent`` remains available as the lower-resolution
    reference guide.  Tensor fields are direct references to native-upscaler
    outputs; the handoff does not clone the large allocations.
    """

    version: int
    source_handoff: DFRTemporalHandoff
    upscaled_video_latent: torch.Tensor
    upscaled_carry_keyframes: torch.Tensor
    upscaled_user_keyframes: tuple[DFRTemporalUserKeyframe, ...]
    keyframe_strength: float


def _expected_spatial_x2_shape(samples: torch.Tensor) -> tuple[int, ...]:
    if not torch.is_tensor(samples) or samples.ndim != 5:
        raise ValueError("Expected a [B,C,T,H,W] latent tensor.")
    batch, channels, frames, height, width = (int(value) for value in samples.shape)
    return (
        batch,
        channels,
        frames,
        height * POST_TEMPORAL_SPATIAL_SCALE,
        width * POST_TEMPORAL_SPATIAL_SCALE,
    )


def _resolve_native_ltxv_latent_upsampler_node_class():
    try:
        import nodes as comfy_nodes  # type: ignore
    except Exception as exc:  # pragma: no cover - pure-torch tests inject a stub
        raise ValueError(
            "Could not import ComfyUI's global nodes registry. Post-temporal Spatial x2 "
            "must run inside ComfyUI with LTXVLatentUpsampler available."
        ) from exc

    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mappings, dict):
        raise ValueError("ComfyUI nodes registry has no NODE_CLASS_MAPPINGS dict.")
    node_cls = mappings.get("LTXVLatentUpsampler")
    if node_cls is None:
        raise ValueError("Native node 'LTXVLatentUpsampler' is not registered.")
    return node_cls


def _extract_native_latent(result: Any, node_name: str) -> dict[str, Any]:
    candidate = result
    if not isinstance(candidate, (dict, tuple, list)) and hasattr(candidate, "result"):
        candidate = candidate.result
    if isinstance(candidate, dict) and "result" in candidate and "samples" not in candidate:
        candidate = candidate["result"]
    if isinstance(candidate, (tuple, list)):
        if not candidate:
            raise ValueError(f"{node_name} returned an empty result sequence.")
        candidate = candidate[0]
    if not isinstance(candidate, dict) or not torch.is_tensor(candidate.get("samples")):
        raise ValueError(f"{node_name} did not return a LATENT dict with tensor 'samples'.")
    return candidate


def _run_native_spatial_upsampler(
    samples: torch.Tensor,
    upscale_model: Any,
    vae: Any,
) -> tuple[torch.Tensor, str]:
    node_cls = _resolve_native_ltxv_latent_upsampler_node_class()
    node = node_cls()
    declared = getattr(node_cls, "FUNCTION", None) or getattr(node, "FUNCTION", None)
    candidates = (declared,) if declared else ("execute", "upscale", "upsample_latent")
    func = None
    for name in candidates:
        if name and callable(getattr(node, name, None)):
            func = getattr(node, name)
            break
    if func is None:
        raise ValueError(f"Native node '{node_cls.__name__}' exposes no supported execution function.")

    # Comfy's native node shallow-copies the LATENT wrapper and replaces
    # ``samples``.  Passing this one-key wrapper avoids cloning the source tensor.
    result = func({"samples": samples}, upscale_model, vae)
    latent = _extract_native_latent(result, node_cls.__name__)
    return latent["samples"], str(node_cls.__name__)


def _validate_spatial_x2_output(
    output: torch.Tensor,
    source: torch.Tensor,
    *,
    label: str,
) -> None:
    expected = _expected_spatial_x2_shape(source)
    actual = tuple(int(value) for value in output.shape)
    if actual != expected:
        raise ValueError(
            f"{label} has shape {actual}, expected spatial x2 shape {expected}. "
            "Make sure the spatial LTX-2.5 latent upscaler is loaded, not the temporal upscaler."
        )


def require_post_temporal_spatial_upscaled_handoff(
    value: Any,
) -> DFRPostTemporalSpatialUpscaledHandoff:
    if not isinstance(value, DFRPostTemporalSpatialUpscaledHandoff):
        raise ValueError(
            f"Expected DFRPostTemporalSpatialUpscaledHandoff, got {type(value).__name__}."
        )
    if int(value.version) != POST_TEMPORAL_SPATIAL_UPSCALED_VERSION:
        raise ValueError(f"Unsupported post-temporal Spatial-upscale version {value.version}.")

    source = require_temporal_handoff(value.source_handoff)
    if int(source.completed_rounds) < 1:
        raise ValueError("Post-temporal Spatial x2 requires at least one completed temporal round.")
    if source.post_temporal_spatial_completed:
        raise ValueError("Post-temporal Spatial x2 has already completed for this handoff.")
    if not source.last_window_seams:
        raise ValueError("Post-temporal Spatial x2 requires the last temporal window seams.")

    _validate_spatial_x2_output(
        value.upscaled_video_latent,
        source.video_latent,
        label="Upscaled post-temporal video",
    )
    _validate_spatial_x2_output(
        value.upscaled_carry_keyframes,
        source.carry_keyframes,
        label="Upscaled post-temporal carry keyframes",
    )
    if int(value.upscaled_carry_keyframes.shape[2]) != len(source.carry_positions):
        raise ValueError("Spatial x2 must preserve the complete carry-keyframe count.")

    if len(value.upscaled_user_keyframes) != len(source.user_keyframes):
        raise ValueError("Spatial x2 must preserve every captured user keyframe.")
    for index, (upscaled, original) in enumerate(
        zip(value.upscaled_user_keyframes, source.user_keyframes)
    ):
        if not isinstance(upscaled, DFRTemporalUserKeyframe):
            raise ValueError(
                f"upscaled_user_keyframes[{index}] has unexpected type {type(upscaled).__name__}."
            )
        if int(upscaled.pixel_frame_index) != int(original.pixel_frame_index):
            raise ValueError("Spatial x2 must preserve user-keyframe source indices.")
        if float(upscaled.strength) != float(original.strength):
            raise ValueError("Spatial x2 must preserve each user-keyframe strength.")
        _validate_spatial_x2_output(
            upscaled.latent,
            original.latent,
            label=f"Upscaled user keyframe {index}",
        )

    strength = float(value.keyframe_strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"keyframe_strength must be in [0,1], got {value.keyframe_strength}.")
    return value


def prepare_post_temporal_spatial_upscale(
    temporal_handoff: DFRTemporalHandoff,
    upscale_model: Any,
    vae: Any,
    *,
    keyframe_strength: float = OFFICIAL_EPILOGUE_KEYFRAME_STRENGTH,
) -> DFRPostTemporalSpatialUpscaledHandoff:
    """Spatially upscale the final temporal latent and every latent guide.

    The complete carry bag is kept: original/refined temporal anchors and newly
    generated midpoint keyframes are never reduced to ``last_window_seams``.
    Those seams remain unchanged in ``source_handoff`` for the later tile plan.
    """
    source = require_temporal_handoff(temporal_handoff)
    if int(source.completed_rounds) < 1:
        raise ValueError("Post-temporal Spatial x2 requires at least one completed temporal round.")
    if source.post_temporal_spatial_completed:
        raise ValueError("Post-temporal Spatial x2 has already completed for this handoff.")

    strength = float(keyframe_strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"keyframe_strength must be in [0,1], got {keyframe_strength}.")

    upscaled_video, _native_node_name = _run_native_spatial_upsampler(
        source.video_latent,
        upscale_model,
        vae,
    )
    _validate_spatial_x2_output(
        upscaled_video,
        source.video_latent,
        label="Upscaled post-temporal video",
    )

    upscaled_carry, _ = _run_native_spatial_upsampler(
        source.carry_keyframes,
        upscale_model,
        vae,
    )
    _validate_spatial_x2_output(
        upscaled_carry,
        source.carry_keyframes,
        label="Upscaled post-temporal carry keyframes",
    )

    upscaled_users: tuple[DFRTemporalUserKeyframe, ...] = ()
    if source.user_keyframes:
        user_stack = torch.cat(
            [keyframe.latent for keyframe in source.user_keyframes],
            dim=2,
        )
        upscaled_user_stack, _ = _run_native_spatial_upsampler(
            user_stack,
            upscale_model,
            vae,
        )
        _validate_spatial_x2_output(
            upscaled_user_stack,
            user_stack,
            label="Upscaled post-temporal user-keyframe stack",
        )
        upscaled_users = tuple(
            DFRTemporalUserKeyframe(
                pixel_frame_index=int(original.pixel_frame_index),
                strength=float(original.strength),
                latent=upscaled_user_stack[:, :, index : index + 1],
            )
            for index, original in enumerate(source.user_keyframes)
        )

    return require_post_temporal_spatial_upscaled_handoff(
        DFRPostTemporalSpatialUpscaledHandoff(
            version=POST_TEMPORAL_SPATIAL_UPSCALED_VERSION,
            source_handoff=source,
            upscaled_video_latent=upscaled_video,
            upscaled_carry_keyframes=upscaled_carry,
            upscaled_user_keyframes=upscaled_users,
            keyframe_strength=strength,
        )
    )
