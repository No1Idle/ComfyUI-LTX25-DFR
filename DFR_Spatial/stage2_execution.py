"""Stage-2F real AV denoise / transformer execution bridge.

This is the Stage-2 counterpart of the already validated Stage-1 joint AV
execution bridge:
- input VIDEO is the strict official Stage-2 token-state AFTER Stage-2C + 2E;
- input AUDIO is the strict official Stage-2 audio token-state produced by 2E;
- execution MUST use the Stage-2 detailing MODEL from 2D (the x2 Pixel-Spatial IC-LoRA clone);
- the noised states remain authoritative and are only materialized temporarily into the
  native Comfy/LTXAV execution geometry;
- one real x0 transformer pass is executed and the result is packed back into strict
  official VIDEO/AUDIO states.

Like Stage 1, this module intentionally stops before Euler/post-process.  It only
produces the raw denoised/x0 prediction for one Stage-2 sigma index.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .av_execution import (
    _execute_stage1_av_via_comfy_native_bridge,
    _execute_stage1_av_via_direct_forward,
    _model_looks_like_comfy_patcher,
    validate_stage1_av_denoise,
)
from .av_model_bridge import DFRStage1AVModelInput, materialize_stage1_av_model_input
from .dfr_noiser import NOISER_METADATA_KEY
from .dfr_sigmas import experimental_stage_2_extra_step_sigmas, stage_2_sigmas, validate_sigma_tensor
from .stage2_detailing import (
    DETAILING_CONTRACT_ATTACHMENT,
    OFFICIAL_DETAILING_LORA_BASENAME,
    OFFICIAL_REFERENCE_DOWNSCALE_FACTOR,
)
from .stage2_noiser import STAGE2_NOISER_STAGE, STAGE2_RNG_ORDER
from .latent_state import clone_or_create_official_state, ensure_token_state
from .audio_state import get_audio_official_state


STAGE2_EXECUTION_PHASE = "2F_stage2_av_denoise"


def _get_attachment(model: Any, key: str) -> Any:
    getter = getattr(model, "get_attachment", None)
    if callable(getter):
        return getter(key)
    attachments = getattr(model, "attachments", None)
    if isinstance(attachments, Mapping):
        return attachments.get(key)
    return None


def _require_exact_stage2_sigmas(
    sigmas: torch.Tensor,
    *,
    allow_experimental_extra_step: bool = False,
) -> torch.Tensor:
    expected = experimental_stage_2_extra_step_sigmas() if allow_experimental_extra_step else stage_2_sigmas()
    schedule_name = "experimental_stage_2_extra_step_sigmas" if allow_experimental_extra_step else "stage_2_sigmas"
    passed, max_error, details = validate_sigma_tensor(sigmas, expected, schedule_name)
    if not passed:
        expected_description = (
            "fixed experimental 5-point / 4-transition Stage-2 schedule"
            if allow_experimental_extra_step
            else "exact official 4-point / 3-transition Stage-2 DFR sigma schedule"
        )
        raise ValueError(
            f"Stage 2F requires the {expected_description}. "
            f"{details}; max_error={max_error:.9g}."
        )
    return sigmas.detach().to(dtype=torch.float32)


def _require_stage2_detailing_model(model: Any) -> dict[str, Any]:
    contract = _get_attachment(model, DETAILING_CONTRACT_ATTACHMENT)
    if not isinstance(contract, Mapping):
        raise ValueError(
            "Stage 2F requires the MODEL produced by 'LTX Official: Apply Stage 2 Detailing IC-LoRA'. "
            "The connected MODEL has no Stage-2 detailing contract attachment."
        )

    errors = []
    if contract.get("phase") != "2D_detailing_iclora":
        errors.append(f"phase={contract.get('phase')!r}")
    if str(contract.get("lora_basename", "")) != OFFICIAL_DETAILING_LORA_BASENAME:
        errors.append(f"lora_basename={contract.get('lora_basename')!r}")
    try:
        strength_model = float(contract.get("strength_model", float("nan")))
    except (TypeError, ValueError):
        strength_model = float("nan")
    if not torch.isfinite(torch.tensor(strength_model)):
        errors.append(f"strength_model={contract.get('strength_model')!r}")
    if int(contract.get("reference_downscale_factor", -1)) != OFFICIAL_REFERENCE_DOWNSCALE_FACTOR:
        errors.append(f"reference_downscale_factor={contract.get('reference_downscale_factor')!r}")

    if errors:
        raise ValueError(
            "Connected MODEL does not satisfy the strict Stage-2 detailing contract required for Stage 2F: "
            + "; ".join(errors)
        )
    return dict(contract)


def resolve_stage2_shared_av_seed(
    noised_stage_2_video_state: dict[str, Any],
    noised_stage_2_audio_state: dict[str, Any],
) -> int:
    """Recover the authoritative Stage-2 seed from the Stage-2 AV noiser metadata."""
    v_state = clone_or_create_official_state(noised_stage_2_video_state)
    v_tokens = ensure_token_state(
        v_state,
        target_samples=noised_stage_2_video_state["samples"],
        fps=float(v_state.get("fps") or 24.0),
        scale_factors=tuple(int(x) for x in v_state.get("scale_factors", (8, 32, 32))),
        causal_fix=bool(v_state.get("causal_fix", True)),
    )
    a_state = get_audio_official_state(noised_stage_2_audio_state)
    a_tokens = a_state["token_state"]

    vm = v_tokens.get(NOISER_METADATA_KEY) or {}
    am = a_tokens.get(NOISER_METADATA_KEY) or {}

    if vm.get("stage") != STAGE2_NOISER_STAGE or am.get("stage") != STAGE2_NOISER_STAGE:
        raise ValueError(
            "Stage 2F requires states produced by 'LTX Official: Stage 2 AV Gaussian Noiser'."
        )
    if vm.get("shared_av_rng") is not True or am.get("shared_av_rng") is not True:
        raise ValueError("Stage-2 AV noiser metadata does not declare shared AV RNG.")
    if vm.get("rng_order") != STAGE2_RNG_ORDER or am.get("rng_order") != STAGE2_RNG_ORDER:
        raise ValueError("Stage-2 AV noiser metadata does not declare the required video_then_audio RNG order.")
    if vm.get("rng_continuation") is not True or am.get("rng_continuation") is not True:
        raise ValueError("Stage-2 AV noiser metadata does not declare continued-not-reseeded RNG semantics.")
    if "seed" not in vm or "seed" not in am:
        raise ValueError("Stage-2 AV noiser metadata is missing the authoritative seed.")
    v_seed = int(vm["seed"])
    a_seed = int(am["seed"])
    if v_seed != a_seed:
        raise ValueError(f"Stage-2 video/audio noiser seed mismatch: video={v_seed}, audio={a_seed}.")
    return v_seed


def execute_stage2_av_denoise(
    model: Any,
    positive: Any,
    negative: Any,
    noised_stage_2_video_state: dict[str, Any],
    noised_stage_2_audio_state: dict[str, Any],
    sigmas: torch.Tensor,
    step_index: int,
    cfg_scale: float = 1.0,
    seed: int | None = None,
    *,
    allow_experimental_extra_step: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], DFRStage1AVModelInput, str]:
    """Run one real joint Stage-2 AV transformer/x0 evaluation."""
    sigma_schedule = _require_exact_stage2_sigmas(
        sigmas,
        allow_experimental_extra_step=allow_experimental_extra_step,
    )
    step_index = int(step_index)
    if step_index < 0 or step_index >= sigma_schedule.numel() - 1:
        raise ValueError(f"step_index must be in [0, {sigma_schedule.numel() - 2}], got {step_index}.")
    cfg_scale = float(cfg_scale)

    contract = _require_stage2_detailing_model(model)
    shared_seed = resolve_stage2_shared_av_seed(noised_stage_2_video_state, noised_stage_2_audio_state)
    if seed is not None and int(seed) != shared_seed:
        raise ValueError(
            f"Explicit execution seed {int(seed)} does not match authoritative Stage-2 shared AV noiser seed {shared_seed}. "
            "The strict parity path must use the continued Stage-1 seed."
        )
    seed = shared_seed

    av_model_input = materialize_stage1_av_model_input(noised_stage_2_video_state, noised_stage_2_audio_state)

    if _model_looks_like_comfy_patcher(model):
        denoised_video_state, denoised_audio_state, core_report = _execute_stage1_av_via_comfy_native_bridge(
            model=model,
            positive=positive,
            negative=negative,
            noised_video_state=noised_stage_2_video_state,
            noised_audio_state=noised_stage_2_audio_state,
            sigma_schedule=sigma_schedule,
            step_index=step_index,
            cfg_scale=cfg_scale,
            seed=seed,
            av_model_input=av_model_input,
        )
    else:
        denoised_video_state, denoised_audio_state, core_report = _execute_stage1_av_via_direct_forward(
            model=model,
            positive=positive,
            negative=negative,
            noised_video_state=noised_stage_2_video_state,
            noised_audio_state=noised_stage_2_audio_state,
            sigma_schedule=sigma_schedule,
            step_index=step_index,
            cfg_scale=cfg_scale,
            seed=seed,
            av_model_input=av_model_input,
        )

    sigma_value = float(sigma_schedule[step_index].item())
    sigma_next = float(sigma_schedule[step_index + 1].item())
    report = (
        f"PASS=True; stage=2F_av_denoise; sigma={sigma_value:.9g}; sigma_next={sigma_next:.9g}; "
        f"step_index={step_index}; cfg_scale={cfg_scale:.9g}; seed={int(seed)}; seed_mode=continued_not_reseeded; "
        f"detail_lora={contract.get('lora_basename')}; detail_strength={float(contract.get('strength_model', 0.0)):.9g}; "
        f"reference_downscale_factor={int(contract.get('reference_downscale_factor', -1))}; "
        f"video_token_shape={tuple(int(x) for x in av_model_input.video_latent.shape)}; "
        f"audio_token_shape={tuple(int(x) for x in av_model_input.audio_latent.shape)}; "
        f"total_token_count={int(av_model_input.total_token_count)}; core=({core_report})."
    )
    return denoised_video_state, denoised_audio_state, av_model_input, report


def validate_stage2_av_denoise(
    noised_stage_2_video_state: dict[str, Any],
    noised_stage_2_audio_state: dict[str, Any],
    raw_denoised_stage_2_video_state: dict[str, Any],
    raw_denoised_stage_2_audio_state: dict[str, Any],
) -> dict[str, Any]:
    """Structural validator for the first real Stage-2 AV transformer step."""
    metadata_error = 0.0
    try:
        shared_seed = resolve_stage2_shared_av_seed(noised_stage_2_video_state, noised_stage_2_audio_state)
    except Exception:
        shared_seed = None
        metadata_error = 1.0

    base = validate_stage1_av_denoise(
        noised_video_state=noised_stage_2_video_state,
        noised_audio_state=noised_stage_2_audio_state,
        raw_denoised_video_state=raw_denoised_stage_2_video_state,
        raw_denoised_audio_state=raw_denoised_stage_2_audio_state,
    )

    passed = bool(base["passed"] and metadata_error == 0.0)
    return {
        **base,
        "passed": passed,
        "metadata_error": float(metadata_error),
        "seed": shared_seed,
    }
