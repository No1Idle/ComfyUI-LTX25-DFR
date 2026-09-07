"""One Temporal-DFR x2 latent-upsample round.

Oracle reference: ``DFRPipeline.__call__`` temporal loop in
``packages/ltx-pipelines/src/ltx_pipelines/dfr_pipeline.py``::

    video_latent = self.temporal_upsampler(video_state.latent[:1])
    num_frames = 2 * (num_frames - 1) + 1
    current_fps = 2 * current_fps
    seam_positions = [2 * position for position in carry_positions]
    anchor_keyframes = carry_keyframes

Only the base video latent is passed through the temporal upsampler.  Carried
keyframes are already single-frame latent planes: their tensor values stay
unchanged and only their pixel-frame positions double.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .temporal_handoff import DFRTemporalHandoff, require_temporal_handoff


TEMPORAL_UPSCALED_HANDOFF_VERSION = 1
TEMPORAL_UPSCALE_FACTOR = 2


@dataclass(frozen=True)
class DFRTemporalUpscaledHandoff:
    """Output of one temporal x2 latent-upsample operation."""

    version: int
    round_index: int
    source_handoff: DFRTemporalHandoff
    upscaled_video_latent: torch.Tensor
    anchor_keyframes: torch.Tensor
    seam_positions: tuple[int, ...]
    source_padded_frames: int
    target_padded_frames: int
    target_requested_frames: int
    source_fps: float
    target_fps: float
    conditioning_fps: float


def official_temporal_conditioning_fps(playback_fps: float) -> float:
    """Match the official high-frame-rate conditioning policy."""
    playback_fps = float(playback_fps)
    if playback_fps <= 0.0:
        raise ValueError(f"playback_fps must be positive, got {playback_fps}.")
    return 60.0 if playback_fps > 30.0 else playback_fps


def _resolve_native_ltxv_latent_upsampler_node_class():
    try:
        import nodes as comfy_nodes  # type: ignore
    except Exception as exc:  # pragma: no cover - pure-torch tests inject a stub
        raise ValueError(
            "Could not import ComfyUI's global nodes registry. Temporal upscaling must run "
            "inside ComfyUI with LTXVLatentUpsampler available."
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


def _run_native_temporal_upsampler(
    samples: torch.Tensor,
    upscale_model: Any,
    vae: Any,
) -> tuple[dict[str, Any], str]:
    """Invoke Comfy's native upsampler without cloning the large source latent."""
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

    # The native node shallow-copies this wrapper before replacing ``samples``;
    # the source tensor itself is read-only, so a second full latent allocation is unnecessary.
    result = func({"samples": samples}, upscale_model, vae)
    return _extract_native_latent(result, node_cls.__name__), str(node_cls.__name__)


def _expected_temporal_x2_shape(samples: torch.Tensor) -> tuple[int, ...]:
    if not torch.is_tensor(samples) or samples.ndim != 5:
        raise ValueError("Expected a 5-D latent tensor shaped [B,C,T,H,W].")
    b, c, t, h, w = (int(x) for x in samples.shape)
    if t < 1:
        raise ValueError("Temporal latent length must be at least 1.")
    # Oracle LatentUpsampler doubles temporal cells then removes its first
    # upsampled cell because LTX's causal first latent represents one pixel frame.
    return (b, c, 2 * t - 1, h, w)


def require_temporal_upscaled_handoff(value: Any) -> DFRTemporalUpscaledHandoff:
    if not isinstance(value, DFRTemporalUpscaledHandoff):
        raise ValueError(f"Expected DFRTemporalUpscaledHandoff, got {type(value).__name__}.")
    if int(value.version) != TEMPORAL_UPSCALED_HANDOFF_VERSION:
        raise ValueError(f"Unsupported Temporal-DFR upscaled handoff version {value.version}.")
    source = require_temporal_handoff(value.source_handoff)
    expected_round = int(source.completed_rounds) + 1
    if int(value.round_index) != expected_round:
        raise ValueError(
            f"Temporal upscaled handoff round_index={value.round_index}, expected {expected_round}."
        )
    expected_shape = _expected_temporal_x2_shape(source.video_latent)
    if tuple(int(x) for x in value.upscaled_video_latent.shape) != expected_shape:
        raise ValueError(
            f"Temporal-upscaled video has shape {tuple(value.upscaled_video_latent.shape)}, expected {expected_shape}."
        )
    if value.anchor_keyframes is not source.carry_keyframes:
        raise ValueError("Temporal carried keyframes must be passed through unchanged, not copied/upscaled.")
    expected_seams = tuple(2 * int(position) for position in source.carry_positions)
    if tuple(int(x) for x in value.seam_positions) != expected_seams:
        raise ValueError(f"Temporal seam positions {value.seam_positions} do not match expected {expected_seams}.")
    expected_padded = 2 * (int(source.padded_frames) - 1) + 1
    expected_requested = 2 * (int(source.requested_frames) - 1) + 1
    if int(value.target_padded_frames) != expected_padded:
        raise ValueError(f"target_padded_frames={value.target_padded_frames}, expected {expected_padded}.")
    if int(value.target_requested_frames) != expected_requested:
        raise ValueError(f"target_requested_frames={value.target_requested_frames}, expected {expected_requested}.")
    if float(value.source_fps) != float(source.fps) or float(value.target_fps) != 2.0 * float(source.fps):
        raise ValueError("Temporal x2 fps metadata is inconsistent with the source handoff.")
    expected_conditioning_fps = official_temporal_conditioning_fps(value.target_fps)
    if float(value.conditioning_fps) != expected_conditioning_fps:
        raise ValueError(
            f"conditioning_fps={value.conditioning_fps}, expected {expected_conditioning_fps}."
        )
    return value


def prepare_temporal_latent_upscale(
    temporal_handoff: DFRTemporalHandoff,
    upscale_model: Any,
    vae: Any,
) -> tuple[DFRTemporalUpscaledHandoff, dict[str, Any], str]:
    """Run exactly the oracle's temporal-upsample operation for the next round.

    This function is source-agnostic: ``temporal_handoff`` may have been built
    from Stage 1 (experimental) or Stage 2 (official Temporal-DFR entry point).
    """
    source = require_temporal_handoff(temporal_handoff)
    if source.post_temporal_spatial_completed:
        raise ValueError(
            "Cannot run another Temporal-DFR round after the post-temporal Spatial x2 epilogue."
        )

    upscaled_latent, native_node_name = _run_native_temporal_upsampler(
        source.video_latent,
        upscale_model,
        vae,
    )
    video = upscaled_latent["samples"]
    expected_shape = _expected_temporal_x2_shape(source.video_latent)
    actual_shape = tuple(int(x) for x in video.shape)
    if actual_shape != expected_shape:
        raise ValueError(
            "The supplied LATENT_UPSCALE_MODEL did not perform official temporal x2 geometry. "
            f"Got {actual_shape}, expected {expected_shape}. "
            "Make sure the temporal LTX-2.5 latent upscaler is loaded, not the spatial upscaler."
        )

    target_padded_frames = 2 * (int(source.padded_frames) - 1) + 1
    target_requested_frames = 2 * (int(source.requested_frames) - 1) + 1
    seam_positions = tuple(2 * int(position) for position in source.carry_positions)
    round_index = int(source.completed_rounds) + 1
    target_fps = 2.0 * float(source.fps)

    result = DFRTemporalUpscaledHandoff(
        version=TEMPORAL_UPSCALED_HANDOFF_VERSION,
        round_index=round_index,
        source_handoff=source,
        upscaled_video_latent=video,
        # Oracle: ``anchor_keyframes = carry_keyframes``.  Preserve identity.
        anchor_keyframes=source.carry_keyframes,
        seam_positions=seam_positions,
        source_padded_frames=int(source.padded_frames),
        target_padded_frames=target_padded_frames,
        target_requested_frames=target_requested_frames,
        source_fps=float(source.fps),
        target_fps=target_fps,
        conditioning_fps=official_temporal_conditioning_fps(target_fps),
    )
    result = require_temporal_upscaled_handoff(result)

    report = (
        f"PASS=True; stage=T2_temporal_latent_upscale; round={round_index}; "
        f"source={source.source_stage}; native_node={native_node_name}; "
        f"source_video_shape={tuple(int(x) for x in source.video_latent.shape)}; "
        f"upscaled_video_shape={actual_shape}; keyframes_upsampled=False; "
        f"anchor_keyframes_shape={tuple(int(x) for x in source.carry_keyframes.shape)}; "
        f"source_frames={source.padded_frames}; target_frames={target_padded_frames}; "
        f"source_fps={source.fps:.9g}; target_fps={result.target_fps:.9g}; "
        f"conditioning_fps={result.conditioning_fps:.9g}; "
        f"seam_positions={seam_positions}."
    )
    return result, upscaled_latent, report
