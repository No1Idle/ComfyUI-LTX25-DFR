"""Stage-2I final-output / trim helper for strict Spatial-DFR parity.

After the full Stage-2 AV loop, upstream Spatial DFR does *not* keep the Stage-2
AUDIO result as the final audio output.  Instead it:

1. takes the Stage-2 detailed VIDEO base latent;
2. trims away any padded temporal tail introduced by the DFR working canvas;
3. keeps the original Stage-1 audio latent as the final audio output.

This module packages that boundary explicitly so the Comfy workflow has a clean,
final pair of decode-ready latents.
"""

from __future__ import annotations

from typing import Any

import torch

from .dfr_layout import validate_layout
from .decoder_keyframes import DFRFinalDecodeKeyframes, build_final_decode_keyframes
from .decoder_handoff import DFRStage2DecoderHandoff
from .stage2_handoff import DFRStage2Handoff
from .stage2_result import DFRStage2ResultHandoff, require_stage2_result_handoff
from .stage2_spatial import DFRUpscaledStage1Handoff, require_upscaled_stage1_handoff


def _validate_video_latent(latent: dict[str, Any], name: str) -> torch.Tensor:
    if not isinstance(latent, dict):
        raise ValueError(f"{name} must be a LATENT dict, got {type(latent).__name__}.")
    samples = latent.get("samples")
    if not torch.is_tensor(samples) or samples.ndim != 5:
        raise ValueError(f"{name} must contain [B,C,T,H,W] tensor 'samples'.")
    return samples


def _validate_audio_latent(latent: dict[str, Any], name: str) -> torch.Tensor:
    if not isinstance(latent, dict):
        raise ValueError(f"{name} must be a LATENT dict, got {type(latent).__name__}.")
    samples = latent.get("samples")
    if not torch.is_tensor(samples) or samples.ndim != 4:
        raise ValueError(f"{name} must contain [B,C,T,F] tensor 'samples'.")
    return samples


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return 1.0
    if a.numel() == 0:
        return 0.0
    return float(
        (a.to(device=b.device, dtype=torch.float32) - b.to(dtype=torch.float32)).abs().max().item()
    )


def finalize_stage2_output(
    stage_2_handoff: DFRStage2Handoff,
    stage_2_base_latent: dict[str, Any],
    dfr_layout: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], str]:
    """Create the decode-ready final video/audio latents for strict Spatial DFR."""
    if not isinstance(stage_2_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_2_handoff).__name__}.")

    layout = validate_layout(dfr_layout)
    video = _validate_video_latent(stage_2_base_latent, "stage_2_base_latent")

    expected_padded_t = int(layout["padded_latent_frames"])
    requested_t = int(layout["requested_latent_frames"])
    if int(video.shape[2]) != expected_padded_t:
        raise ValueError(
            "Stage-2 base latent / layout mismatch: the connected Stage-2 base latent has "
            f"T={int(video.shape[2])}, but the layout requires padded_latent_frames={expected_padded_t}."
        )

    trimmed_video = video[:, :, :requested_t].detach().clone()
    final_video_latent = {"samples": trimmed_video}

    final_audio_tensor = stage_2_handoff.stage_1_audio_latent.detach().clone()
    final_audio_latent = {"samples": final_audio_tensor}

    report = (
        f"PASS=True; stage=2I_final_output; official_audio_source=stage1; "
        f"requested_frames={int(layout['requested_frames'])}; padded_frames={int(layout['padded_frames'])}; "
        f"trimmed_pixel_frames={int(layout['padding_frames'])}; requested_latent_frames={requested_t}; "
        f"padded_latent_frames={expected_padded_t}; trimmed_latent_frames={expected_padded_t - requested_t}; "
        f"final_video_shape={tuple(int(x) for x in trimmed_video.shape)}; "
        f"final_audio_shape={tuple(int(x) for x in final_audio_tensor.shape)}; "
        f"audio_latent_trimmed=False; requires_decoded_audio_trim=True; "
        f"padded_audio_duration_seconds={float(stage_2_handoff.duration_seconds):.9g}."
    )
    return final_video_latent, final_audio_latent, report


def finalize_stage2_output_from_upscaled_handoff(
    upscaled_stage_1_handoff: DFRUpscaledStage1Handoff,
    stage_2_base_latent: dict[str, Any],
    dfr_layout: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], str]:
    """Finalize Stage 2 using the single enriched continuation bundle."""
    bundle = require_upscaled_stage1_handoff(upscaled_stage_1_handoff)
    return finalize_stage2_output(bundle.stage_1_handoff, stage_2_base_latent, dfr_layout)


def finalize_stage2_result(
    stage_2_handoff: DFRStage2ResultHandoff,
    dfr_layout: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return the two decode-ready latents from the compact Stage-2 result."""
    handoff = require_stage2_result_handoff(stage_2_handoff)
    layout = validate_layout(dfr_layout)
    video = handoff.padded_video_latent
    expected_padded_t = int(layout["padded_latent_frames"])
    requested_t = int(layout["requested_latent_frames"])
    if int(video.shape[2]) != expected_padded_t:
        raise ValueError(
            "Stage-2 result/layout mismatch: the handoff video has "
            f"T={int(video.shape[2])}, expected padded_latent_frames={expected_padded_t}."
        )

    # contiguous() is zero-copy when no trim is required and allocates only
    # when a shorter temporal view must be materialized for the VAE.
    final_video = video[:, :, :requested_t].contiguous()
    return (
        {"samples": final_video},
        {"samples": handoff.stage_1_audio_latent},
    )


def prepare_official_final_decode(
    stage_2_handoff: DFRStage2ResultHandoff,
    dfr_layout: dict[str, Any],
) -> tuple[DFRStage2DecoderHandoff, DFRFinalDecodeKeyframes]:
    """Expose the decoder continuation and its post-trim keyframe package."""
    handoff = require_stage2_result_handoff(stage_2_handoff)
    final_keyframes, _kept, _dropped, _report = build_final_decode_keyframes(
        handoff.decoder_handoff,
        dfr_layout,
    )
    return handoff.decoder_handoff, final_keyframes


def split_stage1_audio_video(
    stage_1_handoff: DFRStage2Handoff,
    dfr_layout: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return the handoff's trimmed Stage-1 video and preserved audio latents."""
    if not isinstance(stage_1_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_1_handoff).__name__}.")

    layout = validate_layout(dfr_layout)
    video = stage_1_handoff.reserved_half_res_video
    audio = stage_1_handoff.stage_1_audio_latent
    if not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError("stage_1_handoff reserved video must be a [B,C,T,H,W] tensor.")
    if not torch.is_tensor(audio) or audio.ndim != 4:
        raise ValueError("stage_1_handoff audio must be a [B,C,T,F] tensor.")

    expected_padded_t = int(layout["padded_latent_frames"])
    requested_t = int(layout["requested_latent_frames"])
    if int(video.shape[2]) != expected_padded_t:
        raise ValueError(
            "Stage-1 handoff / layout mismatch: the reserved video has "
            f"T={int(video.shape[2])}, but padded_latent_frames={expected_padded_t}."
        )

    return (
        {"samples": video[:, :, :requested_t].detach().clone()},
        {"samples": audio.detach().clone()},
    )


def validate_stage2_final_output(
    stage_2_handoff: DFRStage2Handoff,
    stage_2_base_latent: dict[str, Any],
    final_video_latent: dict[str, Any],
    final_audio_latent: dict[str, Any],
    dfr_layout: dict[str, Any],
) -> dict[str, Any]:
    """Validate that the final packaged outputs match strict Spatial-DFR semantics."""
    if not isinstance(stage_2_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_2_handoff).__name__}.")

    layout = validate_layout(dfr_layout)
    stage2_base = _validate_video_latent(stage_2_base_latent, "stage_2_base_latent")
    final_video = _validate_video_latent(final_video_latent, "final_video_latent")
    final_audio = _validate_audio_latent(final_audio_latent, "final_audio_latent")

    expected_padded_t = int(layout["padded_latent_frames"])
    requested_t = int(layout["requested_latent_frames"])
    if int(stage2_base.shape[2]) != expected_padded_t:
        raise ValueError(
            "Stage-2 base latent / layout mismatch during validation: "
            f"got T={int(stage2_base.shape[2])}, expected padded_latent_frames={expected_padded_t}."
        )

    expected_video = stage2_base[:, :, :requested_t]
    expected_audio = stage_2_handoff.stage_1_audio_latent

    video_trim_error = _max_abs(final_video, expected_video)
    audio_source_error = _max_abs(final_audio, expected_audio)

    final_video_requested_shape_error = 0.0 if int(final_video.shape[2]) == requested_t else 1.0
    final_video_requested_frames = (int(final_video.shape[2]) - 1) * int(layout["temporal_scale"]) + 1
    final_video_framecount_error = 0.0 if int(final_video_requested_frames) == int(layout["requested_frames"]) else 1.0

    passed = (
        video_trim_error == 0.0
        and audio_source_error == 0.0
        and final_video_requested_shape_error == 0.0
        and final_video_framecount_error == 0.0
    )

    return {
        "passed": bool(passed),
        "video_trim_error": float(video_trim_error),
        "audio_source_error": float(audio_source_error),
        "final_video_requested_shape_error": float(final_video_requested_shape_error),
        "final_video_framecount_error": float(final_video_framecount_error),
        "requested_frames": int(layout["requested_frames"]),
        "padded_frames": int(layout["padded_frames"]),
        "trimmed_pixel_frames": int(layout["padding_frames"]),
        "requested_latent_frames": requested_t,
        "padded_latent_frames": expected_padded_t,
        "trimmed_latent_frames": expected_padded_t - requested_t,
        "final_video_shape": tuple(int(x) for x in final_video.shape),
        "final_audio_shape": tuple(int(x) for x in final_audio.shape),
        "duration_seconds": float(stage_2_handoff.duration_seconds),
    }


def _validate_audio_object(audio: dict[str, Any], name: str) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict):
        raise ValueError(f"{name} must be a Comfy AUDIO dict, got {type(audio).__name__}.")
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if not torch.is_tensor(waveform):
        raise ValueError(f"{name} must contain tensor 'waveform'.")
    if waveform.ndim < 1:
        raise ValueError(f"{name} waveform must have at least one dimension, got {tuple(waveform.shape)}.")
    if sample_rate is None:
        raise ValueError(f"{name} must contain 'sample_rate'.")
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"{name} sample_rate must be > 0, got {sample_rate}.")
    return waveform, sample_rate


def trim_stage2_decoded_audio(
    stage_2_handoff: DFRStage2Handoff,
    decoded_stage_1_audio: dict[str, Any],
    dfr_layout: dict[str, Any],
) -> tuple[dict[str, Any], int, float, str]:
    """Trim decoded Stage-1 audio to the final requested video duration.

    This mirrors the upstream DFR finalization order: decode the preserved Stage-1
    audio latent first, then crop the waveform to round(video_seconds * sample_rate).
    For the current parity target temporal upsampling is disabled, so playback_fps
    is exactly the Stage-1/Stage-2 handoff fps.
    """
    if not isinstance(stage_2_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_2_handoff).__name__}.")

    layout = validate_layout(dfr_layout)
    waveform, sample_rate = _validate_audio_object(decoded_stage_1_audio, "decoded_stage_1_audio")

    playback_fps = float(stage_2_handoff.fps)
    if playback_fps <= 0.0:
        raise ValueError(f"stage_2_handoff.fps must be > 0, got {playback_fps}.")

    requested_frames = int(layout["requested_frames"])
    video_seconds = float(requested_frames) / playback_fps
    target_samples = min(int(waveform.shape[-1]), int(round(video_seconds * sample_rate)))

    trimmed = dict(decoded_stage_1_audio)
    trimmed["waveform"] = waveform[..., :target_samples].clone()
    trimmed["sample_rate"] = sample_rate

    report = (
        f"PASS=True; stage=2I_decoded_audio_trim; official_audio_source=stage1; "
        f"requested_frames={requested_frames}; playback_fps={playback_fps:.9g}; "
        f"video_seconds={video_seconds:.9g}; sample_rate={sample_rate}; "
        f"input_audio_samples={int(waveform.shape[-1])}; target_audio_samples={target_samples}; "
        f"trimmed_audio_samples={int(waveform.shape[-1]) - target_samples}; "
        f"output_waveform_shape={tuple(int(x) for x in trimmed['waveform'].shape)}."
    )
    return trimmed, target_samples, video_seconds, report


def trim_stage2_decoded_audio_from_upscaled_handoff(
    upscaled_stage_1_handoff: DFRUpscaledStage1Handoff,
    decoded_stage_1_audio: dict[str, Any],
    dfr_layout: dict[str, Any],
) -> tuple[dict[str, Any], int, float, str]:
    """Trim final audio using timing metadata carried by the enriched handoff."""
    bundle = require_upscaled_stage1_handoff(upscaled_stage_1_handoff)
    return trim_stage2_decoded_audio(bundle.stage_1_handoff, decoded_stage_1_audio, dfr_layout)


def _trim_decoded_audio_for_fps(
    decoded_audio: dict[str, Any],
    dfr_layout: dict[str, Any],
    fps: float,
) -> dict[str, Any]:
    layout = validate_layout(dfr_layout)
    waveform, sample_rate = _validate_audio_object(decoded_audio, "decoded_audio")
    playback_fps = float(fps)
    if playback_fps <= 0.0:
        raise ValueError(f"Output fps must be positive, got {playback_fps}.")
    video_seconds = float(layout["requested_frames"]) / playback_fps
    target_samples = min(int(waveform.shape[-1]), int(round(video_seconds * sample_rate)))
    trimmed = dict(decoded_audio)
    trimmed["waveform"] = waveform[..., :target_samples].clone()
    trimmed["sample_rate"] = sample_rate
    return trimmed


def trim_stage1_decoded_audio(
    stage_1_handoff: DFRStage2Handoff,
    decoded_audio: dict[str, Any],
    dfr_layout: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(stage_1_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_1_handoff).__name__}.")
    return _trim_decoded_audio_for_fps(decoded_audio, dfr_layout, stage_1_handoff.fps)


def trim_stage2_result_decoded_audio(
    stage_2_handoff: DFRStage2ResultHandoff,
    decoded_audio: dict[str, Any],
    dfr_layout: dict[str, Any],
) -> dict[str, Any]:
    handoff = require_stage2_result_handoff(stage_2_handoff)
    return _trim_decoded_audio_for_fps(decoded_audio, dfr_layout, handoff.fps)


def validate_stage2_decoded_audio_trim(
    stage_2_handoff: DFRStage2Handoff,
    decoded_stage_1_audio: dict[str, Any],
    final_audio: dict[str, Any],
    dfr_layout: dict[str, Any],
) -> dict[str, Any]:
    """Validate the final decoded-audio trim exactly against upstream semantics."""
    if not isinstance(stage_2_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_2_handoff).__name__}.")

    layout = validate_layout(dfr_layout)
    input_waveform, input_rate = _validate_audio_object(decoded_stage_1_audio, "decoded_stage_1_audio")
    final_waveform, final_rate = _validate_audio_object(final_audio, "final_audio")

    playback_fps = float(stage_2_handoff.fps)
    requested_frames = int(layout["requested_frames"])
    video_seconds = float(requested_frames) / playback_fps
    expected_samples = min(int(input_waveform.shape[-1]), int(round(video_seconds * input_rate)))
    expected_waveform = input_waveform[..., :expected_samples]

    sample_rate_error = 0.0 if final_rate == input_rate else 1.0
    sample_count_error = 0.0 if int(final_waveform.shape[-1]) == expected_samples else 1.0
    waveform_error = _max_abs(final_waveform, expected_waveform)

    # Upstream uses round(video_seconds * sample_rate), so one audio sample is the
    # quantization limit when expressing the final waveform duration in seconds.
    actual_seconds = float(final_waveform.shape[-1]) / float(final_rate)
    duration_error_seconds = abs(actual_seconds - video_seconds)
    duration_tolerance_seconds = 1.0 / float(final_rate)
    duration_error = 0.0 if duration_error_seconds <= duration_tolerance_seconds + 1e-12 else duration_error_seconds

    passed = (
        sample_rate_error == 0.0
        and sample_count_error == 0.0
        and waveform_error == 0.0
        and duration_error == 0.0
    )

    return {
        "passed": bool(passed),
        "sample_rate_error": float(sample_rate_error),
        "sample_count_error": float(sample_count_error),
        "waveform_error": float(waveform_error),
        "duration_error": float(duration_error),
        "requested_frames": requested_frames,
        "playback_fps": playback_fps,
        "video_seconds": video_seconds,
        "sample_rate": input_rate,
        "input_audio_samples": int(input_waveform.shape[-1]),
        "expected_audio_samples": expected_samples,
        "final_audio_samples": int(final_waveform.shape[-1]),
        "trimmed_audio_samples": int(input_waveform.shape[-1]) - expected_samples,
        "final_audio_seconds": actual_seconds,
        "final_waveform_shape": tuple(int(x) for x in final_waveform.shape),
    }
