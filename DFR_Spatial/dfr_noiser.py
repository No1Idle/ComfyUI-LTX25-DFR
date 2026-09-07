"""Exact Gaussian noiser/state materialization for the Spatial DFR parity path.

Reference implementation:
    Lightricks/LTX-2
    packages/ltx-core/src/ltx_core/components/noisers.py :: GaussianNoiser

Official DFR creates video LatentState tensors in torch.bfloat16 and then applies
GaussianNoiser.  The denoise mask remains float32 and positions remain float32.

This module deliberately stops *before* any transformer/model call.
"""

from __future__ import annotations

from typing import Any

import torch

from .latent_state import OFFICIAL_STATE_KEY, clone_or_create_official_state, get_official_state


OFFICIAL_DFR_STATE_DTYPE = torch.bfloat16
NOISER_METADATA_KEY = "gaussian_noiser"


def _generator_for(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def _sample_official_noise(reference: torch.Tensor, seed: int) -> torch.Tensor:
    """Mirror GaussianNoiser._sample_noise exactly for one isolated state."""
    generator = _generator_for(reference.device, seed)
    return torch.randn(
        *reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def official_gaussian_noiser_formula(
    latent: torch.Tensor,
    clean_latent: torch.Tensor,
    denoise_mask: torch.Tensor,
    noise: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    """Mirror upstream GaussianNoiser.__call__ lines 29-33 exactly."""
    mixed = torch.lerp(latent.float(), noise.float(), float(noise_scale))
    noised = torch.lerp(clean_latent.float(), mixed, denoise_mask)
    return noised.to(latent.dtype)


def materialize_official_gaussian_noised_state(
    official_state_latent: dict[str, Any],
    *,
    seed: int,
    noise_scale: float = 1.0,
) -> dict[str, Any]:
    """Cast the token state to official DFR BF16 and apply GaussianNoiser.

    The Comfy LATENT ``samples`` tensor is intentionally left untouched: the real
    diffusion state includes appended tokens and therefore cannot be represented by
    a normal [B,C,T,H,W] Comfy latent.  The materialized state lives in
    ``_ltx_dfr_official_state['token_state']`` and is consumed by the next bridge.
    """
    noise_scale = float(noise_scale)
    if not 0.0 <= noise_scale <= 1.0:
        raise ValueError(f"noise_scale must be in [0,1], got {noise_scale}.")

    state = clone_or_create_official_state(official_state_latent)
    token_state = state.get("token_state")
    if token_state is None:
        raise ValueError(
            "Official state has no patchified token_state. For DFR Stage 1, apply "
            "VideoGeneratedKeyframeSlots before the Gaussian noiser."
        )
    if token_state.get(NOISER_METADATA_KEY) is not None:
        raise ValueError(
            "This official token state is already Gaussian-noised. Start from the pre-noise "
            "conditioning state instead of applying the noiser twice."
        )

    # Official DFRPipeline hardcodes dtype=torch.bfloat16 for the LatentState.
    latent_before = token_state["latent"].to(dtype=OFFICIAL_DFR_STATE_DTYPE)
    clean = token_state["clean_latent"].to(device=latent_before.device, dtype=OFFICIAL_DFR_STATE_DTYPE)
    mask = token_state["denoise_mask"].to(device=latent_before.device, dtype=torch.float32)

    noise = _sample_official_noise(latent_before, seed=seed)
    noised = official_gaussian_noiser_formula(
        latent=latent_before,
        clean_latent=clean,
        denoise_mask=mask,
        noise=noise,
        noise_scale=noise_scale,
    )

    token_state["latent"] = noised
    token_state["clean_latent"] = clean
    token_state["denoise_mask"] = mask
    # positions and masks retain their official dtypes.
    token_state["positions"] = token_state["positions"].to(dtype=torch.float32)
    if torch.is_tensor(token_state.get("keyframes_mask")):
        token_state["keyframes_mask"] = token_state["keyframes_mask"].to(dtype=torch.float32)

    token_state[NOISER_METADATA_KEY] = {
        "type": "GaussianNoiser",
        "seed": int(seed),
        "noise_scale": noise_scale,
        "state_dtype": "bfloat16",
        "noise_dtype": "bfloat16",
        "device_type": latent_before.device.type,
        "token_shape": tuple(int(x) for x in latent_before.shape),
    }
    state["operations"].append(
        {
            "type": "GaussianNoiser",
            "seed": int(seed),
            "noise_scale": noise_scale,
            "state_dtype": "bfloat16",
        }
    )

    out = official_state_latent.copy()
    out[OFFICIAL_STATE_KEY] = state
    return out


def noiser_stats(official_state_latent: dict[str, Any]) -> dict[str, Any]:
    state = get_official_state(official_state_latent)
    token_state = state.get("token_state")
    if token_state is None:
        raise ValueError("Official state has no token_state.")
    metadata = token_state.get(NOISER_METADATA_KEY)
    if metadata is None:
        raise ValueError("Official state has not been materialized by GaussianNoiser.")

    mask = token_state["denoise_mask"]
    return {
        **metadata,
        "fully_clean_tokens": int((mask == 0).sum().item()),
        "fully_noised_tokens": int((mask == 1).sum().item()),
        "partial_tokens": int(((mask > 0) & (mask < 1)).sum().item()),
        "mask_min": float(mask.min().item()),
        "mask_max": float(mask.max().item()),
    }
