"""Shared-generator Gaussian noising for one Temporal-DFR tile.

The official pipeline owns one GaussianNoiser from Stage 1 onward.  Every
temporal tile consumes that generator in VIDEO-then-AUDIO order.  Frozen audio
still consumes a Gaussian draw: DiffusionStage applies the noiser before it
sets the frozen mask to zero.
"""

from __future__ import annotations

from typing import Any

import torch

if "." in (__package__ or ""):  # Normal Comfy custom-node package import.
    from ..DFR_Spatial.audio_state import (
        AUDIO_OFFICIAL_STATE_KEY,
        DEFAULT_AUDIO_CAUSAL,
        DEFAULT_AUDIO_HOP_LENGTH,
        DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        DEFAULT_AUDIO_SAMPLE_RATE,
        DEFAULT_AUDIO_SHIFT,
        _audio_positions,
        _new_audio_official_state,
        _unpatchify_audio,
    )
    from ..DFR_Spatial.dfr_noiser import (
        NOISER_METADATA_KEY,
        OFFICIAL_DFR_STATE_DTYPE,
        official_gaussian_noiser_formula,
    )
    from ..DFR_Spatial.latent_state import OFFICIAL_STATE_KEY, clone_or_create_official_state, ensure_token_state
    from ..DFR_Spatial.stage2_noiser import _preferred_comfy_device, _randn_like_with_generator
else:  # Standalone temporal regression tests.
    from DFR_Spatial.audio_state import (
        AUDIO_OFFICIAL_STATE_KEY,
        DEFAULT_AUDIO_CAUSAL,
        DEFAULT_AUDIO_HOP_LENGTH,
        DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        DEFAULT_AUDIO_SAMPLE_RATE,
        DEFAULT_AUDIO_SHIFT,
        _audio_positions,
        _new_audio_official_state,
        _unpatchify_audio,
    )
    from DFR_Spatial.dfr_noiser import (
        NOISER_METADATA_KEY,
        OFFICIAL_DFR_STATE_DTYPE,
        official_gaussian_noiser_formula,
    )
    from DFR_Spatial.latent_state import OFFICIAL_STATE_KEY, clone_or_create_official_state, ensure_token_state
    from DFR_Spatial.stage2_noiser import _preferred_comfy_device, _randn_like_with_generator

from .temporal_conditioning import DFRTemporalPreparedTile, require_temporal_prepared_tile
from .temporal_handoff import require_temporal_handoff


TEMPORAL_NOISER_STAGE = "temporal_continued"
TEMPORAL_RNG_ORDER = "video_then_audio"


def temporal_execution_device(prepared: DFRTemporalPreparedTile, device: torch.device | None = None) -> torch.device:
    tile = require_temporal_prepared_tile(prepared)
    fallback = tile.tile_video_latent.device
    return torch.device(device) if device is not None else _preferred_comfy_device(fallback)


def generator_from_temporal_handoff(
    prepared: DFRTemporalPreparedTile,
    device: torch.device,
) -> torch.Generator:
    """Restore the exact Gaussian stream continuation stored by the handoff."""
    tile = require_temporal_prepared_tile(prepared)
    source = require_temporal_handoff(tile.source_plan.source_upscaled_handoff.source_handoff)
    target = torch.device(device)
    if str(source.rng_device_type) != target.type:
        raise ValueError(
            "Temporal RNG continuation was captured for device type "
            f"{source.rng_device_type!r}, but execution selected {target.type!r}."
        )
    generator = torch.Generator(device=target)
    generator.set_state(source.rng_state_before_temporal.detach().cpu())
    return generator


def _prepare_video_tokens(
    prepared: DFRTemporalPreparedTile,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    state = clone_or_create_official_state(prepared.video_state)
    tokens = ensure_token_state(
        state,
        target_samples=prepared.video_state["samples"],
        fps=float(prepared.conditioning_fps),
        scale_factors=tuple(int(x) for x in state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(state.get("causal_fix", True)),
    )
    if tokens.get(NOISER_METADATA_KEY) is not None:
        raise ValueError("Prepared Temporal-DFR tile is already Gaussian-noised.")
    latent = tokens["latent"].to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    clean = tokens["clean_latent"].to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    mask = tokens["denoise_mask"].to(device=device, dtype=torch.float32)
    tokens["positions"] = tokens["positions"].to(device=device, dtype=torch.float32)
    if torch.is_tensor(tokens.get("keyframes_mask")):
        tokens["keyframes_mask"] = tokens["keyframes_mask"].to(device=device, dtype=torch.float32)
    if torch.is_tensor(tokens.get("attention_mask")):
        tokens["attention_mask"] = tokens["attention_mask"].to(device=device)
    return state, tokens, latent, clean, mask


def _prepare_frozen_audio_tokens(
    prepared: DFRTemporalPreparedTile,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    initial = prepared.frozen_audio_latent.to(device=device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    b, c, t, f = (int(x) for x in initial.shape)
    clean = initial.clone()
    frozen_mask = torch.zeros((b, 1, t, 1), device=device, dtype=torch.float32)
    positions = _audio_positions(
        batch=b,
        frames=t,
        sample_rate=DEFAULT_AUDIO_SAMPLE_RATE,
        hop_length=DEFAULT_AUDIO_HOP_LENGTH,
        audio_latent_downsample_factor=DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
        is_causal=DEFAULT_AUDIO_CAUSAL,
        shift=DEFAULT_AUDIO_SHIFT,
        device=device,
    )
    duration_seconds = float(prepared.source_plan.tiles[prepared.tile_index].local_frames) / float(
        prepared.conditioning_fps
    )
    state = _new_audio_official_state(
        base_shape=(b, c, t, f),
        initial_latent=initial,
        clean_latent=clean,
        denoise_mask=frozen_mask,
        positions=positions,
        duration_seconds=duration_seconds,
        video_alignment={
            "pixel_frames": int(prepared.source_plan.tiles[prepared.tile_index].local_frames),
            "fps": float(prepared.conditioning_fps),
            "video_base_shape": tuple(int(x) for x in prepared.tile_video_latent.shape),
            "frozen": True,
        },
        audio_config={
            "sample_rate": DEFAULT_AUDIO_SAMPLE_RATE,
            "hop_length": DEFAULT_AUDIO_HOP_LENGTH,
            "audio_latent_downsample_factor": DEFAULT_AUDIO_LATENT_DOWNSAMPLE_FACTOR,
            "channels": c,
            "mel_bins": f,
            "is_causal": DEFAULT_AUDIO_CAUSAL,
            "shift": DEFAULT_AUDIO_SHIFT,
        },
    )
    tokens = state["token_state"]
    tokens["frozen"] = True
    return state, tokens, tokens["latent"], tokens["clean_latent"], tokens["denoise_mask"]


def materialize_temporal_tile_gaussian_noised_states(
    prepared_tile: DFRTemporalPreparedTile,
    sigma0: float,
    generator: torch.Generator,
    *,
    device: torch.device | None = None,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    """Noise one prepared tile and return its post-draw generator state.

    The caller owns ``generator`` and passes the same instance through every
    tile in the round.  This function never reseeds it.
    """
    prepared = require_temporal_prepared_tile(prepared_tile)
    noise_scale = float(sigma0)
    if not 0.0 < noise_scale <= 1.0:
        raise ValueError(f"Temporal sigma0 must be in (0,1], got {noise_scale}.")
    target = temporal_execution_device(prepared, device)
    if torch.device(generator.device).type != target.type:
        raise ValueError(
            f"Temporal generator uses {generator.device}, but tile tensors target {target}."
        )

    v_state, v_tokens, v_latent, v_clean, v_mask = _prepare_video_tokens(prepared, target)
    a_state, a_tokens, a_latent, a_clean, a_mask = _prepare_frozen_audio_tokens(prepared, target)

    rng_before = generator.get_state().detach().cpu().clone()
    video_noise = _randn_like_with_generator(v_latent, generator)
    # Official DiffusionStage calls GaussianNoiser before setting audio frozen.
    audio_noise = _randn_like_with_generator(a_latent, generator)
    rng_after = generator.get_state().detach().cpu().clone()

    v_noised = official_gaussian_noiser_formula(
        v_latent, v_clean, v_mask, video_noise, noise_scale
    )
    # Frozen audio uses noise_scale=0.0, then receives an all-zero mask.
    a_noised = official_gaussian_noiser_formula(a_latent, a_clean, a_mask, audio_noise, 0.0)

    source = prepared.source_plan.source_upscaled_handoff.source_handoff
    common = {
        "type": "GaussianNoiser",
        "stage": TEMPORAL_NOISER_STAGE,
        "round_index": int(prepared.source_plan.source_upscaled_handoff.round_index),
        "tile_index": int(prepared.tile_index),
        "seed": int(source.seed),
        "state_dtype": "bfloat16",
        "noise_dtype": "bfloat16",
        "device_type": target.type,
        "shared_av_rng": True,
        "rng_order": TEMPORAL_RNG_ORDER,
        "rng_continuation": True,
        "rng_state_before_temporal_tile": rng_before,
        "rng_state_after_temporal_tile": rng_after,
    }

    v_tokens["latent"] = v_noised
    v_tokens["clean_latent"] = v_clean
    v_tokens["denoise_mask"] = v_mask
    v_tokens[NOISER_METADATA_KEY] = {
        **common,
        "modality": "video",
        "noise_scale": noise_scale,
        "token_shape": tuple(int(x) for x in v_latent.shape),
    }
    v_state["operations"].append(
        {
            "type": "TemporalGaussianNoiser",
            "modality": "video",
            "noise_scale": noise_scale,
            "round_index": common["round_index"],
            "tile_index": common["tile_index"],
        }
    )

    a_tokens["latent"] = a_noised
    a_tokens["clean_latent"] = a_clean
    a_tokens["denoise_mask"] = a_mask
    a_tokens[NOISER_METADATA_KEY] = {
        **common,
        "modality": "audio",
        "noise_scale": 0.0,
        "frozen": True,
        "token_shape": tuple(int(x) for x in a_latent.shape),
    }
    a_state["operations"].append(
        {
            "type": "TemporalGaussianNoiser",
            "modality": "audio",
            "noise_scale": 0.0,
            "frozen": True,
            "round_index": common["round_index"],
            "tile_index": common["tile_index"],
        }
    )

    out_video = prepared.video_state.copy()
    out_video[OFFICIAL_STATE_KEY] = v_state
    out_audio = {
        "samples": _unpatchify_audio(
            a_noised,
            channels=int(a_tokens["channels"]),
            mel_bins=int(a_tokens["mel_bins"]),
        ),
        AUDIO_OFFICIAL_STATE_KEY: a_state,
    }
    return out_video, out_audio, rng_after
