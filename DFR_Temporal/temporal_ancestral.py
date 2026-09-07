"""Official rectified-flow Euler-Ancestral step for Temporal DFR.

Oracle sources:

* ``ltx_core.components.diffusion_steps.EulerAncestralDiffusionStep``;
* ``ltx_pipelines.utils.samplers.euler_ancestral_denoising_loop``;
* the Temporal DFR tile loop in ``ltx_pipelines.dfr_pipeline``.

The step-noise generator here is deliberately separate from the shared
Gaussian noiser generator carried by :class:`DFRTemporalHandoff`.
"""

from __future__ import annotations

import torch


TEMPORAL_ANCESTRAL_ETA = 0.5
TEMPORAL_ANCESTRAL_S_NOISE = 1.0


def temporal_ancestral_noise_seed(base_seed: int, round_index: int, tile_index: int) -> int:
    """Exact official per-tile Temporal DFR step-noise seed."""
    round_index = int(round_index)
    tile_index = int(tile_index)
    if round_index < 1:
        raise ValueError(f"Temporal round_index must be >= 1, got {round_index}.")
    if tile_index < 0:
        raise ValueError(f"Temporal tile_index must be >= 0, got {tile_index}.")
    return int(base_seed) + 1000 * round_index + tile_index


def make_temporal_ancestral_generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=torch.device(device)).manual_seed(int(seed))


def draw_temporal_ancestral_noise(
    reference: torch.Tensor,
    generator: torch.Generator,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Mirror upstream ``_get_plain_noise`` without normalization."""
    target = torch.device(device)
    if torch.device(generator.device).type != target.type:
        raise ValueError(
            f"Temporal ancestral generator uses {generator.device}, but the step targets {target}."
        )
    return torch.randn(
        reference.shape,
        generator=generator,
        dtype=reference.dtype,
        device=target,
    )


def temporal_ancestral_token_update(
    current: torch.Tensor,
    raw_denoised: torch.Tensor,
    denoise_mask: torch.Tensor,
    clean_latent: torch.Tensor,
    *,
    sigma: float,
    sigma_next: float,
    noise: torch.Tensor | None,
    eta: float = TEMPORAL_ANCESTRAL_ETA,
    s_noise: float = TEMPORAL_ANCESTRAL_S_NOISE,
) -> torch.Tensor:
    """Apply the official masked rectified-flow Euler-Ancestral update.

    The denoised prediction is mask-corrected before the step. On every
    nonterminal ancestral transition the updated sample is mask-corrected a
    second time after noise injection, which keeps frozen audio exact while
    still consuming its required RNG draw.
    """
    if tuple(raw_denoised.shape) != tuple(current.shape):
        raise RuntimeError(
            "Temporal AV denoised token shape changed during the trajectory: "
            f"got {tuple(raw_denoised.shape)}, expected {tuple(current.shape)}."
        )
    if noise is not None and tuple(noise.shape) != tuple(current.shape):
        raise RuntimeError(
            f"Temporal ancestral noise shape {tuple(noise.shape)} != latent shape {tuple(current.shape)}."
        )

    execution_device = raw_denoised.device
    state_dtype = current.dtype
    mask_f32 = denoise_mask.to(device=execution_device, dtype=torch.float32)
    clean_f32 = clean_latent.to(device=execution_device, dtype=torch.float32)
    denoised_f32 = (
        raw_denoised.to(device=execution_device, dtype=state_dtype).to(torch.float32) * mask_f32
        + clean_f32 * (1.0 - mask_f32)
    )

    sigma_f32 = torch.as_tensor(float(sigma), device=execution_device, dtype=torch.float32)
    sigma_next_f32 = torch.as_tensor(float(sigma_next), device=execution_device, dtype=torch.float32)
    if bool(sigma_next_f32 == 0):
        # Upstream short-circuits the terminal step to the masked x0 prediction
        # and, importantly, draws no step noise for either modality.
        return denoised_f32.to(dtype=state_dtype).contiguous()
    if float(eta) > 0.0 and noise is None:
        raise ValueError("Temporal Euler-Ancestral requires a noise tensor when eta > 0.")
    if bool(sigma_f32 == 0):
        raise ValueError("Current temporal sigma is zero before the terminal transition.")

    x_f32 = current.to(device=execution_device, dtype=torch.float32)
    downstep_ratio = 1.0 + (sigma_next_f32 / sigma_f32 - 1.0) * float(eta)
    sigma_down = sigma_next_f32 * downstep_ratio
    sigma_down_ratio = sigma_down / sigma_f32
    x_next = sigma_down_ratio * x_f32 + (1.0 - sigma_down_ratio) * denoised_f32

    if float(eta) > 0.0:
        alpha_next = 1.0 - sigma_next_f32
        alpha_down = 1.0 - sigma_down
        renoise_coeff = (
            sigma_next_f32**2 - sigma_down**2 * alpha_next**2 / alpha_down**2
        ).clamp(min=0) ** 0.5
        x_next = (
            (alpha_next / alpha_down) * x_next
            + noise.to(device=execution_device, dtype=torch.float32) * float(s_noise) * renoise_coeff
        )
        x_next = x_next * mask_f32 + clean_f32 * (1.0 - mask_f32)

    return x_next.to(dtype=state_dtype).contiguous()
