"""Stage-2C conditioning assembly for strict Spatial-DFR parity.

Current upstream DFR builds Stage-2 video conditioning in this order:

1. Re-encode/re-apply the ordinary user image conditionings at Stage-2 resolution.
   - frame_idx == 0 -> VideoConditionByLatentIndex
   - frame_idx > 0  -> VideoConditionByKeyframeIndex
2. Append VideoGeneratedKeyframeSlots using the *upsampled Stage-1 generated
   keyframes* as ``initial_keyframes``.
3. Append VideoConditionByReferenceLatent using the reserved Stage-1
   half-resolution video as the clean IC-LoRA reference.

This module implements steps (2) and (3) on top of a LATENT that has already had
step (1) applied through the same official conditioning nodes used in Stage 1.
No model call, LoRA application, noising, or Euler step occurs here.
"""

from __future__ import annotations

from typing import Any

import torch

from .dfr_layout import validate_layout
from .latent_state import (
    OFFICIAL_STATE_KEY,
    apply_video_condition_by_reference_latent,
    apply_video_generated_keyframe_slots,
    get_official_state,
)
from .stage2_handoff import DFRStage2Handoff
from .stage2_detailing import DFRDetailingSpec, OFFICIAL_REFERENCE_DOWNSCALE_FACTOR
from .stage2_spatial import DFRUpscaledStage1Handoff, require_upscaled_stage1_handoff


# Stage 2C must use the factor resolved from the selected detailing LoRA metadata.
DFR_REFERENCE_TEMPORAL_SCALE_FACTOR = 1
DFR_REFERENCE_STRENGTH = 1.0


_ALLOWED_USER_OPS = {
    "VideoConditionByLatentIndex",
    "VideoConditionByKeyframeIndex",
}


def _validate_latent(latent: dict[str, Any], name: str) -> torch.Tensor:
    if not isinstance(latent, dict):
        raise ValueError(f"{name} must be a LATENT dict, got {type(latent).__name__}.")
    samples = latent.get("samples")
    if not torch.is_tensor(samples) or samples.ndim != 5:
        raise ValueError(f"{name} must contain [B,C,T,H,W] tensor 'samples'.")
    return samples


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


def _tensor_or_none_error(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None and b is None:
        return 0.0
    if (a is None) != (b is None):
        return 1.0
    assert a is not None and b is not None
    return _max_abs(a, b)


def _resolve_user_operations(video_after_user_conditions: dict[str, Any]) -> list[dict[str, Any]]:
    state = video_after_user_conditions.get(OFFICIAL_STATE_KEY)
    if state is None:
        return []
    if not isinstance(state, dict):
        raise ValueError(f"Malformed {OFFICIAL_STATE_KEY}: expected dict.")

    operations = list(state.get("operations", []))
    for op in operations:
        op_type = op.get("type")
        if op_type not in _ALLOWED_USER_OPS:
            raise ValueError(
                "Stage 2C expects its video input to contain only ordinary user image/keyframe "
                "conditionings before DFR slots/reference are appended. "
                f"Found prior operation '{op_type}'."
            )
        if op_type == "VideoConditionByLatentIndex" and int(op.get("latent_idx", -1)) != 0:
            raise ValueError(
                "Official combined_image_conditionings uses VideoConditionByLatentIndex only for frame 0. "
                f"Found latent_idx={op.get('latent_idx')}."
            )
        if op_type == "VideoConditionByKeyframeIndex":
            if int(op.get("frame_idx", 0)) <= 0:
                raise ValueError(
                    "Official combined_image_conditionings uses VideoConditionByKeyframeIndex only for "
                    f"non-zero frame indices. Found frame_idx={op.get('frame_idx')}."
                )
    return operations


def _validate_stage2_geometry(
    *,
    stage_2_handoff: DFRStage2Handoff,
    video_after_user_conditions: dict[str, Any],
    upscaled_generated_keyframes: dict[str, Any],
    reserved_half_res_video: dict[str, Any],
    dfr_layout: dict[str, Any],
    detailing_spec: DFRDetailingSpec,
    verify_handoff_reference: bool = True,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if not isinstance(stage_2_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_2_handoff).__name__}.")
    if not isinstance(detailing_spec, DFRDetailingSpec):
        raise ValueError(f"Expected DFRDetailingSpec, got {type(detailing_spec).__name__}.")
    reference_factor = int(detailing_spec.reference_downscale_factor)
    if reference_factor != OFFICIAL_REFERENCE_DOWNSCALE_FACTOR:
        raise ValueError(
            "Current Spatial x2 DFR requires detailing metadata reference_downscale_factor=2; "
            f"resolved {reference_factor}."
        )

    layout = validate_layout(dfr_layout)
    target = _validate_latent(video_after_user_conditions, "video_after_user_conditions")
    slots = _validate_latent(upscaled_generated_keyframes, "upscaled_generated_keyframes")
    reference = _validate_latent(reserved_half_res_video, "reserved_half_res_video")
    user_ops = _resolve_user_operations(video_after_user_conditions)

    target_pixel_frames = (int(target.shape[2]) - 1) * int(layout["temporal_scale"]) + 1
    if target_pixel_frames != int(layout["padded_frames"]):
        raise ValueError(
            "Stage-2 target/layout mismatch: target latent represents "
            f"{target_pixel_frames} pixel frames, layout requires {layout['padded_frames']}."
        )

    expected_k = len(layout["pixel_frame_indices"])
    expected_slots_shape = (
        int(target.shape[0]),
        int(target.shape[1]),
        expected_k,
        int(target.shape[3]),
        int(target.shape[4]),
    )
    if tuple(slots.shape) != expected_slots_shape:
        raise ValueError(
            "Stage-2 generated-slot seed shape mismatch: expected "
            f"{expected_slots_shape}, got {tuple(slots.shape)}."
        )

    expected_reference_shape = (
        int(target.shape[0]),
        int(target.shape[1]),
        int(target.shape[2]),
        int(target.shape[3]) // reference_factor,
        int(target.shape[4]) // reference_factor,
    )
    if target.shape[3] % reference_factor or target.shape[4] % reference_factor:
        raise ValueError(
            "Stage-2 target latent spatial dimensions must be divisible by the x2 reference factor. "
            f"Got HxW={target.shape[3]}x{target.shape[4]}."
        )
    if tuple(reference.shape) != expected_reference_shape:
        raise ValueError(
            "Reserved Stage-1 reference must be exactly half the Stage-2 spatial latent resolution "
            "with matching batch/channels/time. Expected "
            f"{expected_reference_shape}, got {tuple(reference.shape)}."
        )

    # Phase 2A is the authority for the reserved Stage-1 reference. Prove the
    # connected tensor is still exactly that handoff tensor.
    if (
        verify_handoff_reference
        and _max_abs(reference, stage_2_handoff.reserved_half_res_video) != 0.0
    ):
        raise ValueError(
            "reserved_half_res_video no longer matches the Stage-2A handoff exactly."
        )

    # A non-zero user keyframe in Stage 2 must be encoded at full Stage-2
    # spatial resolution. The operation records the conditioning shape.
    target_hw = (int(target.shape[3]), int(target.shape[4]))
    for op in user_ops:
        if op.get("type") == "VideoConditionByKeyframeIndex":
            shape = tuple(int(x) for x in op.get("conditioning_shape", ()))
            if len(shape) != 5 or shape[-2:] != target_hw:
                raise ValueError(
                    "Stage-2 user keyframe was not encoded at the full Stage-2 spatial resolution. "
                    f"Recorded conditioning_shape={shape}, target HxW={target_hw}."
                )

    return layout, target, slots, reference, user_ops


def assemble_stage2_conditioning(
    stage_2_handoff: DFRStage2Handoff,
    video_after_user_conditions: dict[str, Any],
    upscaled_generated_keyframes: dict[str, Any],
    reserved_half_res_video: dict[str, Any],
    dfr_layout: dict[str, Any],
    detailing_spec: DFRDetailingSpec,
    *,
    verify_handoff_reference: bool = True,
    collect_diagnostics: bool = True,
) -> tuple[dict[str, Any], str]:
    """Append seeded DFR slots then the half-res IC-LoRA reference exactly in upstream order."""
    layout, target, slots, reference, user_ops = _validate_stage2_geometry(
        stage_2_handoff=stage_2_handoff,
        video_after_user_conditions=video_after_user_conditions,
        upscaled_generated_keyframes=upscaled_generated_keyframes,
        reserved_half_res_video=reserved_half_res_video,
        dfr_layout=dfr_layout,
        detailing_spec=detailing_spec,
        verify_handoff_reference=verify_handoff_reference,
    )

    fps = float(stage_2_handoff.fps)
    reference_factor = int(detailing_spec.reference_downscale_factor)
    positions = [int(x) for x in layout["pixel_frame_indices"]]

    with_slots = apply_video_generated_keyframe_slots(
        target_latent=video_after_user_conditions,
        pixel_frame_indices=positions,
        initial_keyframes=upscaled_generated_keyframes,
        fps=fps,
    )
    stage_2_video_state = apply_video_condition_by_reference_latent(
        target_latent=with_slots,
        reference_latent=reserved_half_res_video,
        downscale_factor=reference_factor,
        temporal_scale_factor=DFR_REFERENCE_TEMPORAL_SCALE_FACTOR,
        strength=DFR_REFERENCE_STRENGTH,
        fps=fps,
    )

    if not collect_diagnostics:
        return stage_2_video_state, ""

    state = get_official_state(stage_2_video_state)
    token_state = state["token_state"]
    slot_layout = token_state["generated_keyframe_layout"]
    reference_op = state["operations"][-1]

    user_keyframe_tokens = sum(
        int(op.get("appended_token_count", 0))
        for op in user_ops
        if op.get("type") == "VideoConditionByKeyframeIndex"
    )
    report = (
        f"PASS=True; stage=2C_conditioning; fps={fps:.9g}; "
        f"user_conditionings={len(user_ops)}; user_keyframe_tokens={user_keyframe_tokens}; "
        f"generated_positions={','.join(str(x) for x in positions)}; "
        f"seeded_slots=True; slot_tokens={int(slot_layout['num_tokens'])}; "
        f"reference_downscale_factor={reference_factor}; "
        f"reference_strength={DFR_REFERENCE_STRENGTH:.9g}; "
        f"reference_tokens={int(reference_op['appended_token_count'])}; "
        f"target_shape={tuple(int(x) for x in target.shape)}; "
        f"slot_seed_shape={tuple(int(x) for x in slots.shape)}; "
        f"reference_shape={tuple(int(x) for x in reference.shape)}; "
        f"base_tokens={int(token_state['base_token_count'])}; total_tokens={int(token_state['latent'].shape[1])}; "
        f"operation_order={'->'.join(str(op.get('type')) for op in state['operations'])}."
    )
    return stage_2_video_state, report


def assemble_stage2_conditioning_from_handoff(
    stage_2_handoff: DFRStage2Handoff,
    video_after_user_conditions: dict[str, Any],
    upscaled_generated_keyframes: dict[str, Any],
    dfr_layout: dict[str, Any],
    detailing_spec: DFRDetailingSpec,
) -> tuple[dict[str, Any], str]:
    """Build Stage 2C using its existing handoff as the reserved reference authority."""
    if not isinstance(stage_2_handoff, DFRStage2Handoff):
        raise ValueError(f"Expected DFRStage2Handoff, got {type(stage_2_handoff).__name__}.")

    return assemble_stage2_conditioning(
        stage_2_handoff=stage_2_handoff,
        detailing_spec=detailing_spec,
        video_after_user_conditions=video_after_user_conditions,
        upscaled_generated_keyframes=upscaled_generated_keyframes,
        reserved_half_res_video={"samples": stage_2_handoff.reserved_half_res_video},
        dfr_layout=dfr_layout,
    )


def assemble_stage2_conditioning_from_upscaled_handoff(
    upscaled_stage_1_handoff: DFRUpscaledStage1Handoff,
    video_after_user_conditions: dict[str, Any],
    dfr_layout: dict[str, Any],
    detailing_spec: DFRDetailingSpec,
) -> dict[str, Any]:
    """Finish Stage-2 conditioning without exposing internal carried tensors."""
    bundle = require_upscaled_stage1_handoff(upscaled_stage_1_handoff)
    stage_2_video_state, _report = assemble_stage2_conditioning(
        stage_2_handoff=bundle.stage_1_handoff,
        detailing_spec=detailing_spec,
        video_after_user_conditions=video_after_user_conditions,
        upscaled_generated_keyframes={"samples": bundle.upscaled_generated_keyframes},
        reserved_half_res_video={"samples": bundle.stage_1_handoff.reserved_half_res_video},
        dfr_layout=dfr_layout,
        verify_handoff_reference=False,
        collect_diagnostics=False,
    )
    return stage_2_video_state


def _compare_token_states(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, float]:
    return {
        "latent": _max_abs(actual["latent"], expected["latent"]),
        "clean": _max_abs(actual["clean_latent"], expected["clean_latent"]),
        "mask": _max_abs(actual["denoise_mask"], expected["denoise_mask"]),
        "positions": _max_abs(actual["positions"], expected["positions"]),
        "keyframes_mask": _tensor_or_none_error(actual.get("keyframes_mask"), expected.get("keyframes_mask")),
        "attention_mask": _tensor_or_none_error(actual.get("attention_mask"), expected.get("attention_mask")),
    }


def validate_stage2_conditioning(
    stage_2_handoff: DFRStage2Handoff,
    stage_2_video_state: dict[str, Any],
    video_after_user_conditions: dict[str, Any],
    upscaled_generated_keyframes: dict[str, Any],
    reserved_half_res_video: dict[str, Any],
    dfr_layout: dict[str, Any],
    detailing_spec: DFRDetailingSpec,
) -> dict[str, Any]:
    layout, target, slots, reference, user_ops = _validate_stage2_geometry(
        stage_2_handoff=stage_2_handoff,
        video_after_user_conditions=video_after_user_conditions,
        upscaled_generated_keyframes=upscaled_generated_keyframes,
        reserved_half_res_video=reserved_half_res_video,
        dfr_layout=dfr_layout,
        detailing_spec=detailing_spec,
    )
    fps = float(stage_2_handoff.fps)
    reference_factor = int(detailing_spec.reference_downscale_factor)
    positions = [int(x) for x in layout["pixel_frame_indices"]]

    expected_with_slots = apply_video_generated_keyframe_slots(
        target_latent=video_after_user_conditions,
        pixel_frame_indices=positions,
        initial_keyframes=upscaled_generated_keyframes,
        fps=fps,
    )
    expected = apply_video_condition_by_reference_latent(
        target_latent=expected_with_slots,
        reference_latent=reserved_half_res_video,
        downscale_factor=reference_factor,
        temporal_scale_factor=DFR_REFERENCE_TEMPORAL_SCALE_FACTOR,
        strength=DFR_REFERENCE_STRENGTH,
        fps=fps,
    )

    actual_state = get_official_state(stage_2_video_state)
    expected_state = get_official_state(expected)
    actual_tokens = actual_state["token_state"]
    expected_tokens = expected_state["token_state"]

    sample_error = _max_abs(stage_2_video_state["samples"], target)
    base_clean_error = _max_abs(actual_state["clean_latent"], expected_state["clean_latent"])
    base_mask_error = _max_abs(actual_state["denoise_mask"], expected_state["denoise_mask"])
    token_errors = _compare_token_states(actual_tokens, expected_tokens)
    state_error = max(sample_error, base_clean_error, base_mask_error, *token_errors.values())

    expected_layout = expected_tokens["generated_keyframe_layout"]
    actual_layout = actual_tokens.get("generated_keyframe_layout")
    layout_error = 0.0 if actual_layout == expected_layout else 1.0

    actual_ops = actual_state.get("operations", [])
    expected_ops = expected_state.get("operations", [])
    ordering_error = 0.0 if actual_ops == expected_ops else 1.0

    # Directly validate the important deferred Stage-2 case: initial_keyframes
    # must occupy the generated-slot *current latent* range exactly.
    slot_start = int(expected_layout["first_token"])
    slot_count = int(expected_layout["num_tokens"])
    slot_stop = slot_start + slot_count
    expected_slot_tokens = slots.permute(0, 2, 3, 4, 1).reshape(slots.shape[0], -1, slots.shape[1]).to(
        device=actual_tokens["latent"].device,
        dtype=actual_tokens["latent"].dtype,
    )
    slot_seed_error = _max_abs(actual_tokens["latent"][:, slot_start:slot_stop], expected_slot_tokens)
    slot_clean_zero_error = float(actual_tokens["clean_latent"][:, slot_start:slot_stop].abs().max().item())
    slot_mask_error = float((actual_tokens["denoise_mask"][:, slot_start:slot_stop] - 1.0).abs().max().item())
    slot_keyframe_mask_error = float(
        (actual_tokens["keyframes_mask"][:, slot_start:slot_stop] - 1.0).abs().max().item()
    )

    # Direct reference-tail checks.
    ref_op = actual_ops[-1] if actual_ops else {}
    if ref_op.get("type") != "VideoConditionByReferenceLatent":
        reference_error = 1.0
        reference_mask_error = 1.0
        reference_keyframe_mask_error = 1.0
    else:
        ref_start = int(ref_op["appended_token_start"])
        ref_count = int(ref_op["appended_token_count"])
        ref_stop = ref_start + ref_count
        expected_ref_tokens = reference.permute(0, 2, 3, 4, 1).reshape(
            reference.shape[0], -1, reference.shape[1]
        ).to(device=actual_tokens["clean_latent"].device, dtype=actual_tokens["clean_latent"].dtype)
        reference_error = _max_abs(actual_tokens["clean_latent"][:, ref_start:ref_stop], expected_ref_tokens)
        reference_mask_error = float(actual_tokens["denoise_mask"][:, ref_start:ref_stop].abs().max().item())
        reference_keyframe_mask_error = float(actual_tokens["keyframes_mask"][:, ref_start:ref_stop].abs().max().item())
        reference_error = max(
            reference_error,
            float(actual_tokens["latent"][:, ref_start:ref_stop].abs().max().item()),
        )

    passed = bool(
        state_error == 0.0
        and layout_error == 0.0
        and ordering_error == 0.0
        and slot_seed_error == 0.0
        and slot_clean_zero_error == 0.0
        and slot_mask_error == 0.0
        and slot_keyframe_mask_error == 0.0
        and reference_error == 0.0
        and reference_mask_error == 0.0
        and reference_keyframe_mask_error == 0.0
    )

    return {
        "passed": passed,
        "state_error": state_error,
        "slot_seed_error": slot_seed_error,
        "reference_error": reference_error,
        "layout_error": layout_error,
        "ordering_error": ordering_error,
        "slot_clean_zero_error": slot_clean_zero_error,
        "slot_mask_error": slot_mask_error,
        "slot_keyframe_mask_error": slot_keyframe_mask_error,
        "reference_mask_error": reference_mask_error,
        "reference_keyframe_mask_error": reference_keyframe_mask_error,
        "sample_error": sample_error,
        "base_clean_error": base_clean_error,
        "base_mask_error": base_mask_error,
        "token_latent_error": token_errors["latent"],
        "token_clean_error": token_errors["clean"],
        "token_mask_error": token_errors["mask"],
        "token_position_error": token_errors["positions"],
        "token_keyframes_mask_error": token_errors["keyframes_mask"],
        "token_attention_mask_error": token_errors["attention_mask"],
        "user_conditionings": len(user_ops),
        "positions": tuple(positions),
        "reference_downscale_factor": reference_factor,
        "target_shape": tuple(int(x) for x in target.shape),
        "slot_shape": tuple(int(x) for x in slots.shape),
        "reference_shape": tuple(int(x) for x in reference.shape),
        "total_tokens": int(actual_tokens["latent"].shape[1]),
        "operation_order": tuple(str(op.get("type")) for op in actual_ops),
    }
