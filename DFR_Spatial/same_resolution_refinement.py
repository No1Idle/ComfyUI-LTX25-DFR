"""Experimental ComfyUI adaptation: same-resolution DFR refinement without IC-LoRA.

Reuses the ported Lightricks AV noiser, sampler and decoder handoff. This path
changes conditioning and model requirements; it does not claim upstream parity.
"""
import torch

from .av_execution import (
    _execute_stage1_av_whole_schedule_comfy,
    _execute_stage1_av_whole_schedule_direct,
    _model_looks_like_comfy_patcher,
)
from .decoder_handoff import prepare_stage2_decoder_handoff
from .dfr_layout import validate_layout
from .dfr_execution import extract_official_base_latent
from .dfr_sigmas import stage_2_sigmas, validate_custom_sigma_schedule
from .latent_state import apply_video_generated_keyframe_slots
from .stage2_conditioning import _resolve_user_operations
from .stage2_detailing import DETAILING_CONTRACT_ATTACHMENT
from .stage2_execution import _get_attachment, resolve_stage2_shared_av_seed
from .stage2_handoff import DFRStage2Handoff, HANDOFF_VERSION
from .stage2_noiser import materialize_stage2_av_gaussian_noised_states
from .stage2_result import (
    DFRStage2ResultHandoff, STAGE2_RESULT_HANDOFF_VERSION, require_stage2_result_handoff,
)


def require_stage1(handoff):
    if not isinstance(handoff, DFRStage2Handoff) or handoff.version != HANDOFF_VERSION:
        raise ValueError("Expected a supported Stage 1 handoff.")
    for name in ("reserved_half_res_video", "stage_1_generated_keyframes"):
        value = getattr(handoff, name)
        if not torch.is_tensor(value) or value.ndim != 5:
            raise ValueError(f"Stage 1 {name} must be a five-dimensional latent.")
    return handoff


def extract_stage1_video(handoff):
    # Keep downstream conditioning edits isolated from the reusable Stage-1 state.
    return {"samples": require_stage1(handoff).reserved_half_res_video.clone()}


def prepare_refinement_conditioning(handoff, video_after_user_conditions, dfr_layout):
    handoff = require_stage1(handoff)
    layout = validate_layout(dfr_layout)
    video = video_after_user_conditions.get("samples")
    if not torch.is_tensor(video) or tuple(video.shape) != tuple(handoff.reserved_half_res_video.shape):
        raise ValueError("Refinement video must retain Stage 1 batch, channels, frames, height and width.")
    if (video.shape[2] - 1) * layout["temporal_scale"] + 1 != layout["padded_frames"]:
        raise ValueError("Refinement layout does not match the Stage 1 video frame count.")
    positions = layout["pixel_frame_indices"]
    expected = (video.shape[0], video.shape[1], len(positions), video.shape[3], video.shape[4])
    if tuple(handoff.stage_1_generated_keyframes.shape) != expected:
        raise ValueError("Stage 1 generated keyframes do not match the supplied layout and video geometry.")
    _resolve_user_operations(video_after_user_conditions)
    return apply_video_generated_keyframe_slots(
        target_latent=video_after_user_conditions,
        pixel_frame_indices=list(positions),
        initial_keyframes={"samples": handoff.stage_1_generated_keyframes},
        fps=handoff.fps,
    )


def run_same_resolution_refinement(model, positive, stage_1_handoff,
                                   video_after_user_conditions, dfr_layout,
                                   negative=None, sigmas=None, cfg_scale=1.0):
    handoff = require_stage1(stage_1_handoff)
    if _get_attachment(model, DETAILING_CONTRACT_ATTACHMENT) is not None:
        raise ValueError("Connect the original base MODEL, not Prepare Stage 2 Detailing Model's output.")
    schedule = validate_custom_sigma_schedule(
        stage_2_sigmas() if sigmas is None else sigmas, "Same-resolution refinement sigmas"
    )
    if float(schedule[0]) > 1:
        raise ValueError("Refinement starting sigma must be at most 1.")
    layout = dict(validate_layout(dfr_layout))
    state = prepare_refinement_conditioning(handoff, video_after_user_conditions, layout)
    video, audio, _ = materialize_stage2_av_gaussian_noised_states(
        stage_2_handoff=handoff, stage_2_video_state=state,
        stage_1_audio_for_stage2={"samples": handoff.stage_1_audio_latent},
        stage_2_sigmas_tensor=schedule, require_official_sigmas=False,
        require_reference=False, collect_diagnostics=False,
    )
    seed = resolve_stage2_shared_av_seed(video, audio)
    backend = (_execute_stage1_av_whole_schedule_comfy if _model_looks_like_comfy_patcher(model)
               else _execute_stage1_av_whole_schedule_direct)
    final_video, final_audio, _, _ = backend(
        model=model, positive=positive, negative=negative,
        noised_video_state=video, noised_audio_state=audio,
        sigma_schedule=schedule, cfg_scale=float(cfg_scale), seed=seed,
        collect_diagnostics=False, profiler=None,
    )
    decoder, _, _ = prepare_stage2_decoder_handoff(final_video, final_audio, collect_diagnostics=False)
    return require_stage2_result_handoff(DFRStage2ResultHandoff(
        version=STAGE2_RESULT_HANDOFF_VERSION,
        padded_video_latent=extract_official_base_latent(final_video)["samples"],
        stage_1_audio_latent=handoff.stage_1_audio_latent,
        fps=handoff.fps, duration_seconds=handoff.duration_seconds,
        decoder_handoff=decoder,
        sigma_schedule=schedule.detach().to(device="cpu", dtype=torch.float32).clone(),
        used_official_sigmas=torch.equal(schedule.cpu(), stage_2_sigmas()),
        dfr_layout=layout,
        temporal_seams=tuple(getattr(handoff, "temporal_seams", ())),
    ))


class LTXExtractStage1VideoLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"stage_1_handoff": ("LTX_DFR_STAGE2_HANDOFF",)}}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("video_latent",)
    FUNCTION = "extract"
    CATEGORY = "LTX/DFR Spatial/05 - Experimental Refinement"
    DESCRIPTION = "Extract a copy of Stage 1 video at its original size. Reapply user image conditioning before refinement."

    def extract(self, stage_1_handoff):
        return (extract_stage1_video(stage_1_handoff),)


class LTXRunSameResolutionRefinement:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Original base model, without the detailing IC-LoRA."}),
                "positive": ("CONDITIONING",),
                "stage_1_handoff": ("LTX_DFR_STAGE2_HANDOFF",),
                "video_after_user_conditions": ("LATENT",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "sigmas": ("SIGMAS", {"tooltip": "Defaults to 0.909375, 0.725, 0.421875, 0. Lower the starting sigma for gentler refinement."}),
            },
        }

    RETURN_TYPES = ("LTX_DFR_STAGE2_RESULT_HANDOFF",)
    RETURN_NAMES = ("stage_2_handoff",)
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/05 - Experimental Refinement"
    DESCRIPTION = (
        "Experimental second AV pass at Stage 1 resolution. Internally appends and refines the generated "
        "keyframes; no upscaler, detailing LoRA or reference video. Connect user-conditioned extracted video "
        "directly, without Finalize Stage 2 DFR Conditioning. Preserves original Stage 1 output audio, "
        "matching the standard Stage 2 result; refined video/keyframes feed the existing decoder."
    )

    def run(self, **kwargs):
        return (run_same_resolution_refinement(**kwargs),)
