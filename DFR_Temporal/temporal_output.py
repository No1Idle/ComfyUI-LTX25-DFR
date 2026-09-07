"""Final output preparation and keyframe-aware DiffVAE decode for Temporal DFR.

The official pipeline trims the padded temporal canvas only after the final x2
round, drops carry-forward keyframes outside that requested canvas, restores the
shared Gaussian generator at its post-temporal-noising continuation point, and
then invokes the same keyframe-aware video decoder used by Spatial DFR.  Audio
continues to come from the preserved Stage-1 latent and is cropped only after
the audio VAE decode.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch

if "." in (__package__ or ""):  # Normal Comfy custom-node package import.
    from ..DFR_Spatial.decoder_handoff import (
        DECODER_HANDOFF_VERSION,
        DFRStage2DecoderHandoff,
    )
    from ..DFR_Spatial.decoder_keyframes import DFRFinalDecodeKeyframes, build_final_decode_keyframes
    from ..DFR_Spatial.decoder_pipeline import _decode_keyframe_aware_video
    from ..DFR_Spatial.dfr_layout import make_custom_generated_slot_layout
else:  # Standalone temporal regression tests.
    from DFR_Spatial.decoder_handoff import (
        DECODER_HANDOFF_VERSION,
        DFRStage2DecoderHandoff,
    )
    from DFR_Spatial.decoder_keyframes import DFRFinalDecodeKeyframes, build_final_decode_keyframes
    from DFR_Spatial.decoder_pipeline import _decode_keyframe_aware_video
    from DFR_Spatial.dfr_layout import make_custom_generated_slot_layout

from .temporal_handoff import DFRTemporalHandoff, require_temporal_handoff


def _release_models_before_large_terminal_decode(handoff: DFRTemporalHandoff) -> None:
    """Evict completed sampling models before a post-Spatial DiffVAE decode.

    Comfy's VAE preparation estimates the decoder model load, but not the full
    keyframe-aware Stage-4 feature plus tiled Stage-5 accumulation peak.  After
    the terminal Spatial epilogue that can leave the transformer/LoRA and both
    latent upsamplers resident even though no later sampling node can use them.
    """
    if not bool(handoff.post_temporal_spatial_completed):
        return
    try:
        model_management = importlib.import_module("comfy.model_management")
    except Exception:  # pragma: no cover - standalone regression environment
        return
    unload_all = getattr(model_management, "unload_all_models", None)
    if callable(unload_all):
        unload_all()
    empty_cache = getattr(model_management, "soft_empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def _requested_latent_frames(handoff: DFRTemporalHandoff) -> int:
    return (int(handoff.requested_frames) - 1) // int(handoff.temporal_scale) + 1


def _temporal_decode_log_stage(handoff: DFRTemporalHandoff) -> str:
    temporal_factor = 2 ** int(handoff.completed_rounds)
    stage = f"Temporal x{temporal_factor}"
    if handoff.post_temporal_spatial_completed:
        stage += " + Spatial x2"
    return stage


def finalize_temporal_output(
    temporal_handoff: DFRTemporalHandoff,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float]:
    """Trim final video to the requested canvas and expose preserved audio.

    ``contiguous`` is zero-copy when the requested and padded canvases match;
    when a tail trim is required it materializes only the smaller decode input.
    The audio latent is intentionally not trimmed before the audio VAE decode.
    """
    handoff = require_temporal_handoff(temporal_handoff)
    requested_t = _requested_latent_frames(handoff)
    if requested_t > int(handoff.video_latent.shape[2]):
        raise ValueError(
            f"Requested temporal latent length {requested_t} exceeds padded length "
            f"{int(handoff.video_latent.shape[2])}."
        )
    final_video = handoff.video_latent[:, :, :requested_t].contiguous()
    return (
        {"samples": final_video},
        {"samples": handoff.stage_1_audio_latent},
        float(handoff.fps),
    )


def _validate_final_video_latent(
    handoff: DFRTemporalHandoff,
    final_video_latent: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(final_video_latent, dict):
        raise ValueError(
            f"final_video_latent must be a LATENT dict, got {type(final_video_latent).__name__}."
        )
    samples = final_video_latent.get("samples")
    if not torch.is_tensor(samples) or samples.ndim != 5:
        raise ValueError("final_video_latent must contain a [B,C,T,H,W] tensor named 'samples'.")
    expected = handoff.video_latent[:, :, : _requested_latent_frames(handoff)]
    if tuple(samples.shape) != tuple(expected.shape):
        raise ValueError(
            f"final_video_latent shape {tuple(samples.shape)} does not match the requested Temporal-DFR "
            f"slice {tuple(expected.shape)}."
        )
    if samples.device != expected.device or samples.dtype != expected.dtype:
        raise ValueError(
            "final_video_latent must retain the temporal handoff tensor device and dtype: "
            f"got {samples.device}/{samples.dtype}, expected {expected.device}/{expected.dtype}."
        )
    return final_video_latent


def prepare_temporal_decoder_inputs(
    temporal_handoff: DFRTemporalHandoff,
    final_video_latent: dict[str, Any],
) -> tuple[
    dict[str, Any],
    DFRStage2DecoderHandoff,
    DFRFinalDecodeKeyframes,
    dict[str, Any],
]:
    """Adapt the final Temporal handoff to the validated Spatial decoder core.

    Upstream ``decode_keyframes_from_slots`` drops carry planes at or beyond the
    trimmed frame count.  The Spatial decoder core validates a compact layout,
    so this adapter first performs that exact filter and presents only the
    surviving planes/positions to the shared decoder implementation.
    """
    handoff = require_temporal_handoff(temporal_handoff)
    final_video_latent = _validate_final_video_latent(handoff, final_video_latent)

    source_positions = tuple(int(position) for position in handoff.carry_positions)
    kept_source_indices = tuple(
        index for index, position in enumerate(source_positions) if 0 <= position < int(handoff.requested_frames)
    )
    kept_positions = tuple(source_positions[index] for index in kept_source_indices)
    if not kept_positions:
        raise ValueError(
            "Temporal final decode has no carry-forward keyframe inside the requested output canvas."
        )

    if kept_source_indices == tuple(range(len(source_positions))):
        kept_latents = handoff.carry_keyframes
    else:
        kept_latents = torch.cat(
            [handoff.carry_keyframes[:, :, index : index + 1] for index in kept_source_indices],
            dim=2,
        )

    decoder_handoff = DFRStage2DecoderHandoff(
        version=DECODER_HANDOFF_VERSION,
        seed=int(handoff.seed),
        # After a completed round this is the shared Gaussian generator after
        # every tile's VIDEO-then-AUDIO initial-noise draws. Official ancestral
        # step noise uses separate per-tile generators and does not alter it.
        rng_state_after_stage2_av=handoff.rng_state_before_temporal.detach().cpu().clone(),
        rng_device_type=str(handoff.rng_device_type),
        stage_2_generated_keyframes=kept_latents,
        generated_keyframe_count=int(kept_latents.shape[2]),
        generated_keyframe_shape=tuple(int(value) for value in kept_latents.shape),
    )
    decode_layout = make_custom_generated_slot_layout(
        int(handoff.requested_frames),
        kept_positions,
        int(handoff.temporal_scale),
    )
    final_keyframes, _kept, _dropped, _report = build_final_decode_keyframes(
        decoder_handoff,
        decode_layout,
    )
    return final_video_latent, decoder_handoff, final_keyframes, decode_layout


def decode_temporal_dfr_video(  # noqa: PLR0913
    temporal_handoff: DFRTemporalHandoff,
    final_video_latent: dict[str, Any],
    vae: Any,
    vae_name: str,
    *,
    use_auto_tiling: bool = True,
    tile_frames: int = 104,
    tile_height: int = 416,
    tile_width: int = 544,
) -> torch.Tensor:
    """Run the validated keyframe-aware DiffVAE on a final temporal result."""
    handoff = require_temporal_handoff(temporal_handoff)
    final_video_latent, decoder_handoff, final_keyframes, decode_layout = prepare_temporal_decoder_inputs(
        handoff,
        final_video_latent,
    )
    _release_models_before_large_terminal_decode(handoff)
    return _decode_keyframe_aware_video(
        final_video_latent,
        decoder_handoff,
        final_keyframes,
        decode_layout,
        vae,
        str(vae_name),
        use_auto_tiling=bool(use_auto_tiling),
        tile_frames=int(tile_frames),
        tile_height=int(tile_height),
        tile_width=int(tile_width),
        log_stage=_temporal_decode_log_stage(handoff),
    )


def trim_temporal_decoded_audio(
    temporal_handoff: DFRTemporalHandoff,
    decoded_audio: dict[str, Any],
) -> dict[str, Any]:
    """Apply the official post-audio-VAE crop for the final temporal FPS."""
    handoff = require_temporal_handoff(temporal_handoff)
    if not isinstance(decoded_audio, dict):
        raise ValueError(f"decoded_audio must be a Comfy AUDIO dict, got {type(decoded_audio).__name__}.")
    waveform = decoded_audio.get("waveform")
    if not torch.is_tensor(waveform) or waveform.ndim < 1:
        raise ValueError("decoded_audio must contain a waveform tensor.")
    sample_rate = int(decoded_audio.get("sample_rate", 0))
    if sample_rate <= 0:
        raise ValueError(f"decoded_audio sample_rate must be positive, got {sample_rate}.")

    video_seconds = float(handoff.requested_frames) / float(handoff.fps)
    target_samples = min(int(waveform.shape[-1]), int(round(video_seconds * sample_rate)))
    final_audio = dict(decoded_audio)
    final_audio["waveform"] = waveform[..., :target_samples].clone()
    final_audio["sample_rate"] = sample_rate
    return final_audio
