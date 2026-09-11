"""Stage-2B exact x2 latent spatial upscaling for strict Spatial-DFR parity.

Upstream DFR performs the Stage-2 boundary in three conceptually separate parts:

1. freeze Stage-1 outputs + shared RNG continuation state (Stage 2A);
2. run the dedicated LTX latent spatial upscaler on:
   - the reserved half-resolution Stage-1 base video, and
   - the generated keyframe-slot stack;
3. feed those upscaled tensors into the Stage-2 detailing pass while keeping
   Stage-1 audio as the audio initialization.

This module implements part (2) while validating that the supplied inputs still
match the Stage-2A handoff exactly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

from .stage2_handoff import DFRStage2Handoff, HANDOFF_VERSION


SPATIAL_UPSCALE_FACTOR = 2
UPSCALED_STAGE1_HANDOFF_VERSION = 1


@dataclass(frozen=True)
class DFRUpscaledStage1Handoff:
    """Stage-1 continuation data plus both tensors produced by the x2 upscaler.

    The tensor fields intentionally share storage with the public LATENT output
    and downstream LATENT wrappers.  The handoff is a wiring bundle, not a
    second copy of the large GPU allocations.
    """

    version: int
    stage_1_handoff: DFRStage2Handoff
    upscaled_video_latent: torch.Tensor
    upscaled_generated_keyframes: torch.Tensor


def require_upscaled_stage1_handoff(value: Any) -> DFRUpscaledStage1Handoff:
    if not isinstance(value, DFRUpscaledStage1Handoff):
        raise ValueError(f"Expected DFRUpscaledStage1Handoff, got {type(value).__name__}.")
    if int(value.version) != UPSCALED_STAGE1_HANDOFF_VERSION:
        raise ValueError(f"Unsupported upscaled Stage-1 handoff version {value.version}.")
    if not isinstance(value.stage_1_handoff, DFRStage2Handoff):
        raise ValueError("Upscaled Stage-1 handoff does not contain a valid Stage-1 handoff.")
    if not torch.is_tensor(value.upscaled_video_latent) or value.upscaled_video_latent.ndim != 5:
        raise ValueError("Upscaled Stage-1 video must be a [B,C,T,H,W] tensor.")
    if not torch.is_tensor(value.upscaled_generated_keyframes) or value.upscaled_generated_keyframes.ndim != 5:
        raise ValueError("Upscaled Stage-1 generated keyframes must be a [B,C,K,H,W] tensor.")
    return value


def _clone_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)


def clone_latent_dict(latent: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(latent, dict):
        raise ValueError(f"Expected LATENT dict, got {type(latent).__name__}.")
    samples = latent.get("samples")
    if not torch.is_tensor(samples):
        raise ValueError("LATENT dict must contain a tensor under 'samples'.")
    return {key: _clone_value(value) for key, value in latent.items()}


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return 1.0
    if a.numel() == 0:
        return 0.0
    return float(
        (a.to(device=b.device, dtype=torch.float32) - b.to(dtype=torch.float32))
        .abs()
        .max()
        .item()
    )


def _expected_x2_shape(samples: torch.Tensor) -> tuple[int, ...]:
    if not torch.is_tensor(samples) or samples.ndim != 5:
        raise ValueError("Expected a 5-D latent tensor shaped [B,C,T,H,W].")
    b, c, t, h, w = (int(x) for x in samples.shape)
    return (b, c, t, h * SPATIAL_UPSCALE_FACTOR, w * SPATIAL_UPSCALE_FACTOR)


def _resolve_native_ltxv_latent_upsampler_node_class():
    try:
        import nodes as comfy_nodes  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in pure-torch tests via stub injection
        raise ValueError(
            "Could not import ComfyUI's global nodes registry. "
            "This Stage-2B node must run inside ComfyUI with LTXVLatentUpsampler available."
        ) from exc

    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if not isinstance(mappings, dict):
        raise ValueError("ComfyUI nodes registry has no NODE_CLASS_MAPPINGS dict.")

    node_cls = mappings.get("LTXVLatentUpsampler")
    if node_cls is None:
        raise ValueError(
            "Native node 'LTXVLatentUpsampler' is not registered. "
            "Install/load ComfyUI-LTXVideo (or the native LTX support that provides it)."
        )
    return node_cls


def _extract_latent_from_node_result(result: Any, node_name: str) -> dict[str, Any]:
    """Normalize legacy tuple/dict results and current Comfy V3 IO.NodeOutput."""
    candidate = result

    # ComfyUI's current V3 nodes return comfy_api.latest.IO.NodeOutput.  Its
    # public ``result`` property is the tuple of node output values.  Older
    # Comfy nodes returned tuples directly, so support both without importing
    # a particular comfy_api version here.
    if not isinstance(candidate, (dict, tuple, list)) and hasattr(candidate, "result"):
        candidate = candidate.result

    # Also accept the legacy execution-style wrapper {"result": (...), "ui": ...}.
    if isinstance(candidate, dict) and "result" in candidate and "samples" not in candidate:
        candidate = candidate["result"]

    if isinstance(candidate, (tuple, list)):
        if not candidate:
            raise ValueError(f"{node_name} returned an empty result sequence.")
        candidate = candidate[0]

    if not isinstance(candidate, dict):
        raise ValueError(f"{node_name} returned unexpected type {type(candidate).__name__}.")
    samples = candidate.get("samples")
    if not torch.is_tensor(samples):
        raise ValueError(f"{node_name} did not return a LATENT dict with tensor 'samples'.")
    return candidate


def run_native_ltxv_latent_upsampler(
    latent: dict[str, Any],
    upscale_model: Any,
    vae: Any,
) -> tuple[dict[str, Any], str]:
    node_cls = _resolve_native_ltxv_latent_upsampler_node_class()
    node = node_cls()
    function_name = getattr(node_cls, "FUNCTION", None) or getattr(node, "FUNCTION", None) or "upscale"
    func = getattr(node, function_name, None)
    if func is None:
        raise ValueError(
            f"Native node '{node_cls.__name__}' does not expose callable function '{function_name}'."
        )
    result = func(clone_latent_dict(latent), upscale_model, vae)
    out = _extract_latent_from_node_result(result, node_cls.__name__)
    return out, str(node_cls.__name__)


def _validate_stage2b_inputs_against_handoff(
    stage_2_handoff: DFRStage2Handoff,
    reserved_half_res_video: dict[str, Any],
    stage_1_audio_latent: dict[str, Any],
    stage_1_generated_keyframes: dict[str, Any],
) -> dict[str, float]:
    if not isinstance(stage_2_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_2_handoff).__name__}.")

    rv = reserved_half_res_video.get("samples")
    au = stage_1_audio_latent.get("samples")
    kf = stage_1_generated_keyframes.get("samples")
    if not torch.is_tensor(rv):
        raise ValueError("reserved_half_res_video must contain tensor 'samples'.")
    if not torch.is_tensor(au):
        raise ValueError("stage_1_audio_latent must contain tensor 'samples'.")
    if not torch.is_tensor(kf):
        raise ValueError("stage_1_generated_keyframes must contain tensor 'samples'.")

    return {
        "input_video_error": _max_abs(rv, stage_2_handoff.reserved_half_res_video),
        "input_audio_error": _max_abs(au, stage_2_handoff.stage_1_audio_latent),
        "input_keyframes_error": _max_abs(kf, stage_2_handoff.stage_1_generated_keyframes),
    }


def prepare_stage2_spatial_upscale(
    stage_2_handoff: DFRStage2Handoff,
    reserved_half_res_video: dict[str, Any],
    stage_1_audio_latent: dict[str, Any],
    stage_1_generated_keyframes: dict[str, Any],
    upscale_model: Any,
    vae: Any,
) -> tuple[DFRStage2Handoff, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Run the exact native x2 latent upscaler on Stage-1 video + keyframe outputs."""
    input_errors = _validate_stage2b_inputs_against_handoff(
        stage_2_handoff,
        reserved_half_res_video,
        stage_1_audio_latent,
        stage_1_generated_keyframes,
    )
    if any(value != 0.0 for value in input_errors.values()):
        raise ValueError(
            "Stage-2B inputs do not exactly match the Stage-2A handoff outputs. "
            f"input_video_error={input_errors['input_video_error']}; "
            f"input_audio_error={input_errors['input_audio_error']}; "
            f"input_keyframes_error={input_errors['input_keyframes_error']}."
        )

    upscaled_video_latent, native_node_name = run_native_ltxv_latent_upsampler(
        reserved_half_res_video,
        upscale_model,
        vae,
    )
    upscaled_generated_keyframes, _ = run_native_ltxv_latent_upsampler(
        stage_1_generated_keyframes,
        upscale_model,
        vae,
    )
    stage_1_audio_for_stage2 = clone_latent_dict(stage_1_audio_latent)

    expected_video_shape = _expected_x2_shape(reserved_half_res_video["samples"])
    expected_keyframes_shape = _expected_x2_shape(stage_1_generated_keyframes["samples"])
    actual_video_shape = tuple(int(x) for x in upscaled_video_latent["samples"].shape)
    actual_keyframes_shape = tuple(int(x) for x in upscaled_generated_keyframes["samples"].shape)
    if actual_video_shape != expected_video_shape:
        raise ValueError(
            f"Upscaled video latent has shape {actual_video_shape}, expected exact x2 shape {expected_video_shape}."
        )
    if actual_keyframes_shape != expected_keyframes_shape:
        raise ValueError(
            "Upscaled generated keyframes have shape "
            f"{actual_keyframes_shape}, expected exact x2 shape {expected_keyframes_shape}."
        )

    report = (
        f"PASS=True; stage=2B_spatial_upscale; scale_factor={SPATIAL_UPSCALE_FACTOR}; "
        f"native_node={native_node_name}; seed={int(stage_2_handoff.seed)}; "
        f"reserved_half_res_video_shape={tuple(int(x) for x in reserved_half_res_video['samples'].shape)}; "
        f"upscaled_video_latent_shape={actual_video_shape}; "
        f"stage_1_generated_keyframes_shape={tuple(int(x) for x in stage_1_generated_keyframes['samples'].shape)}; "
        f"upscaled_generated_keyframes_shape={actual_keyframes_shape}; "
        f"stage_1_audio_shape={tuple(int(x) for x in stage_1_audio_for_stage2['samples'].shape)}."
    )
    return (
        stage_2_handoff,
        upscaled_video_latent,
        stage_1_audio_for_stage2,
        upscaled_generated_keyframes,
        report,
    )


def prepare_stage2_spatial_upscale_from_stage2_result_for_test(
    stage_2_handoff: Any,
    upscale_model: Any,
    vae: Any,
) -> tuple[DFRUpscaledStage1Handoff, dict[str, Any]]:
    """TEST ONLY: adapt a completed Stage-2 result into the legacy Stage-1-shaped boundary.

    This deliberately reuses the existing Stage-2 spatial pipeline without introducing
    a new generalized handoff type.  The field names in the synthetic
    :class:`DFRStage2Handoff` are therefore semantically historical: the video and
    generated-keyframe tensors come from completed Stage 2, and the RNG continuation
    is ``rng_state_after_stage2_av``.  Audio intentionally remains the preserved
    Stage-1 audio, matching the existing Spatial/Temporal DFR convention.

    The original Stage-1 noiser token-shape fields are left empty on purpose.  They
    are only required by independent Stage-1 RNG replay validators, whose assumptions
    do not apply to this experimental continuation boundary.
    """
    # Local import avoids the intentional stage2_result -> stage2_spatial dependency
    # becoming a module-level circular import.
    from .stage2_result import require_stage2_result_handoff

    completed = require_stage2_result_handoff(stage_2_handoff)
    decoder = completed.decoder_handoff

    video = completed.padded_video_latent
    keyframes = decoder.stage_2_generated_keyframes
    audio = completed.stage_1_audio_latent
    rng_state = decoder.rng_state_after_stage2_av

    if not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError("Completed Stage-2 video must be [B,C,T,H,W].")
    if not torch.is_tensor(keyframes) or keyframes.ndim != 5:
        raise ValueError("Completed Stage-2 generated keyframes must be [B,C,K,H,W].")
    if not torch.is_tensor(audio) or audio.ndim != 4:
        raise ValueError("Preserved Stage-1 audio must be [B,C,T,F].")
    if not torch.is_tensor(rng_state):
        raise ValueError("Completed Stage-2 handoff has no RNG continuation state.")

    # Mirror the normal Stage-1 handoff rule: only the first video batch continues
    # through DFR.  Tensors are not cloned here; the synthetic handoff is only a
    # wiring adapter and the native x2 upscaler produces the new allocations.
    synthetic = DFRStage2Handoff(
        version=HANDOFF_VERSION,
        seed=int(decoder.seed),
        rng_state_after_stage1_av=rng_state.detach().cpu().clone(),
        rng_device_type=str(decoder.rng_device_type),
        reserved_half_res_video=video[:1],
        stage_1_generated_keyframes=keyframes,
        stage_1_audio_latent=audio,
        video_token_shape_at_stage1_noise=(),
        audio_token_shape_at_stage1_noise=(),
        fps=float(completed.fps),
        duration_seconds=float(completed.duration_seconds),
        dfr_layout=getattr(completed, "dfr_layout", None),
        temporal_seams=tuple(getattr(completed, "temporal_seams", ())),
        temporal_tiles=int(getattr(completed, "temporal_tiles", 1)),
    )

    return prepare_stage2_spatial_upscale_from_handoff(
        synthetic,
        upscale_model,
        vae,
    )


def prepare_stage2_spatial_upscale_from_handoff(
    stage_1_handoff: DFRStage2Handoff,
    upscale_model: Any,
    vae: Any,
) -> tuple[DFRUpscaledStage1Handoff, dict[str, Any]]:
    """Build the lean public Stage-2B bundle from the Stage-1 boundary."""
    if not isinstance(stage_1_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_1_handoff).__name__}.")

    upscaled_video_latent, _native_node_name = run_native_ltxv_latent_upsampler(
        {"samples": stage_1_handoff.reserved_half_res_video},
        upscale_model,
        vae,
    )
    upscaled_generated_keyframes, _ = run_native_ltxv_latent_upsampler(
        {"samples": stage_1_handoff.stage_1_generated_keyframes},
        upscale_model,
        vae,
    )

    video = upscaled_video_latent["samples"]
    keyframes = upscaled_generated_keyframes["samples"]
    expected_video_shape = _expected_x2_shape(stage_1_handoff.reserved_half_res_video)
    expected_keyframes_shape = _expected_x2_shape(stage_1_handoff.stage_1_generated_keyframes)
    if tuple(video.shape) != expected_video_shape:
        raise ValueError(
            f"Upscaled video latent has shape {tuple(video.shape)}, expected exact x2 shape {expected_video_shape}."
        )
    if tuple(keyframes.shape) != expected_keyframes_shape:
        raise ValueError(
            "Upscaled generated keyframes have shape "
            f"{tuple(keyframes.shape)}, expected exact x2 shape {expected_keyframes_shape}."
        )

    upscaled_handoff = DFRUpscaledStage1Handoff(
        version=UPSCALED_STAGE1_HANDOFF_VERSION,
        stage_1_handoff=stage_1_handoff,
        upscaled_video_latent=video,
        upscaled_generated_keyframes=keyframes,
    )
    return upscaled_handoff, upscaled_video_latent


def validate_stage2_spatial_upscale(
    stage_2_handoff: DFRStage2Handoff,
    reserved_half_res_video: dict[str, Any],
    stage_1_audio_latent: dict[str, Any],
    stage_1_generated_keyframes: dict[str, Any],
    upscaled_video_latent: dict[str, Any],
    stage_1_audio_for_stage2: dict[str, Any],
    upscaled_generated_keyframes: dict[str, Any],
    upscale_model: Any,
    vae: Any,
) -> dict[str, Any]:
    input_errors = _validate_stage2b_inputs_against_handoff(
        stage_2_handoff,
        reserved_half_res_video,
        stage_1_audio_latent,
        stage_1_generated_keyframes,
    )

    rerun_video, native_node_name = run_native_ltxv_latent_upsampler(
        reserved_half_res_video,
        upscale_model,
        vae,
    )
    rerun_keyframes, _ = run_native_ltxv_latent_upsampler(
        stage_1_generated_keyframes,
        upscale_model,
        vae,
    )

    video_error = _max_abs(upscaled_video_latent["samples"], rerun_video["samples"])
    keyframes_error = _max_abs(upscaled_generated_keyframes["samples"], rerun_keyframes["samples"])
    audio_error = _max_abs(stage_1_audio_for_stage2["samples"], stage_1_audio_latent["samples"])

    expected_video_shape = _expected_x2_shape(reserved_half_res_video["samples"])
    expected_keyframes_shape = _expected_x2_shape(stage_1_generated_keyframes["samples"])
    actual_video_shape = tuple(int(x) for x in upscaled_video_latent["samples"].shape)
    actual_keyframes_shape = tuple(int(x) for x in upscaled_generated_keyframes["samples"].shape)
    video_shape_error = 0.0 if actual_video_shape == expected_video_shape else 1.0
    keyframes_shape_error = 0.0 if actual_keyframes_shape == expected_keyframes_shape else 1.0

    passed = bool(
        input_errors["input_video_error"] == 0.0
        and input_errors["input_audio_error"] == 0.0
        and input_errors["input_keyframes_error"] == 0.0
        and video_error == 0.0
        and keyframes_error == 0.0
        and audio_error == 0.0
        and video_shape_error == 0.0
        and keyframes_shape_error == 0.0
    )
    return {
        "passed": passed,
        "input_video_error": input_errors["input_video_error"],
        "input_audio_error": input_errors["input_audio_error"],
        "input_keyframes_error": input_errors["input_keyframes_error"],
        "video_error": video_error,
        "keyframes_error": keyframes_error,
        "audio_error": audio_error,
        "video_shape_error": video_shape_error,
        "keyframes_shape_error": keyframes_shape_error,
        "seed": int(stage_2_handoff.seed),
        "native_node": native_node_name,
        "reserved_video_shape": tuple(int(x) for x in reserved_half_res_video["samples"].shape),
        "upscaled_video_shape": actual_video_shape,
        "keyframes_shape": tuple(int(x) for x in stage_1_generated_keyframes["samples"].shape),
        "upscaled_keyframes_shape": actual_keyframes_shape,
        "audio_shape": tuple(int(x) for x in stage_1_audio_for_stage2["samples"].shape),
        "scale_factor": SPATIAL_UPSCALE_FACTOR,
    }
