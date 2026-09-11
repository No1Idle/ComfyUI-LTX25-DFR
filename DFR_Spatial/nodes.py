"""ComfyUI nodes for the strict/native Spatial DFR parity path."""

from __future__ import annotations

from .dfr_layout import layout_from_handoff

import torch

from .dfr_layout import (
    make_custom_generated_slot_layout,
    make_experimental_dense_slot_layout,
    make_layout,
    validate_layout,
)
from .dfr_sigmas import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    stage_1_sigmas,
    stage_2_sigmas,
    validate_sigma_tensor,
)

from .dfr_noiser import (
    NOISER_METADATA_KEY,
    materialize_official_gaussian_noised_state,
    noiser_stats,
)

from .dfr_model_bridge import (
    DFRComfyModelInput,
    materialize_comfy_model_input,
    validate_model_input_roundtrip,
)

from .dfr_execution import (
    euler_step_from_official_denoised,
    execute_comfy_model_denoise,
    execute_stage1_denoising_loop,
    post_process_official_denoised,
    validate_stage1_loop_outputs,
)

from .latent_state import (
    OFFICIAL_STATE_KEY,
    apply_video_condition_by_keyframe_index,
    apply_video_condition_by_latent_index,
    apply_video_condition_by_reference_latent,
    apply_video_generated_keyframe_slots,
    extract_generated_keyframes,
    get_official_state,
)

from .audio_state import (
    create_audio_official_state_from_video,
    audio_state_stats,
    validate_audio_state_against_video,
    materialize_audio_gaussian_noised_state,
    validate_audio_gaussian_noiser,
)

from .av_noiser import (
    materialize_stage1_av_gaussian_noised_states,
    validate_stage1_av_gaussian_noiser,
)

from .av_model_bridge import (
    DFRStage1AVModelInput,
    materialize_stage1_av_model_input,
    av_model_input_stats,
    validate_stage1_av_model_input,
)

from .av_execution import (
    execute_stage1_av_denoise,
    execute_stage1_av_denoising_loop,
    validate_stage1_av_denoise,
    validate_stage1_av_loop_outputs,
)

from .av_step import (
    stage1_av_euler_step_from_denoised,
    validate_stage1_av_euler_step,
)

from .stage1_pipeline import run_stage1_spatial_dfr

from .stage2_handoff import (
    DFRStage2Handoff,
    prepare_stage2_handoff,
    validate_stage2_handoff,
)

from .stage2_spatial import (
    prepare_stage2_spatial_upscale_from_handoff,
    prepare_stage2_spatial_upscale_from_stage2_result_for_test,
    validate_stage2_spatial_upscale,
)

from .stage2_conditioning import (
    assemble_stage2_conditioning_from_upscaled_handoff,
    validate_stage2_conditioning,
)

from .stage2_detailing import (
    OFFICIAL_DETAILING_LORA_BASENAME,
    OFFICIAL_DETAILING_STRENGTH,
    apply_stage2_detailing_lora,
    lora_choices,
    prepare_stage2_detailing_model,
    resolve_stage2_detailing_spec,
    validate_stage2_detailing_lora,
)

from .model_capability import (
    assert_generated_keyframes_supported,
    generated_keyframe_capability,
)

from .stage2_noiser import (
    materialize_stage2_av_gaussian_noised_states,
    validate_stage2_av_gaussian_noiser,
)

from .stage2_execution import (
    execute_stage2_av_denoise,
    validate_stage2_av_denoise,
)

from .stage2_step import (
    stage2_av_euler_step_from_denoised,
    validate_stage2_av_euler_step,
)

from .stage2_loop import (
    execute_experimental_stage2_av_denoising_loop_extra_step_from_upscaled_handoff,
    validate_stage2_av_loop_outputs,
)
from .stage2_pipeline import run_stage2_spatial_dfr

from .conditioning_resize import bilinear_fill_resize_center_crop

from .decoder_handoff import (
    DFRStage2DecoderHandoff,
    prepare_stage2_decoder_handoff,
    validate_stage2_decoder_handoff,
)

from .decoder_keyframes import (
    DFRFinalDecodeKeyframes,
    build_final_decode_keyframes,
    validate_final_decode_keyframes,
)

from .decoder_runtime_probe import probe_decoder_runtime, runtime_probe_report
from .decoder_keyframe_runtime_probe import (
    probe_decoder_keyframe_runtime,
    keyframe_runtime_probe_report,
    keyframe_runtime_supported,
)
from .decoder_keyframe_substrate import (
    DFRDecoderKeyframeSubstrate,
    build_decoder_keyframe_substrate,
    validate_decoder_keyframe_substrate,
    substrate_report,
)
from .decoder_keyframe_checkpoint import (
    DFRDecoderKeyframeCheckpointWeights,
    extract_decoder_keyframe_checkpoint_weights,
    validate_decoder_keyframe_checkpoint_weights,
)
from .decoder_joint_attention import (
    DFRJointDecoderAttention,
    prepare_joint_decoder_attention,
    validate_joint_decoder_attention,
)
from .decoder_det_stage_context import (
    DFRDeterministicDecoderKeyframeContext,
    prepare_deterministic_decoder_keyframe_context,
    validate_deterministic_decoder_keyframe_context,
)
from .decoder_det_stage_official_preflight import (
    DFROfficialDeterministicStagePreflight,
    prepare_official_deterministic_stage_preflight,
    validate_official_deterministic_stage_preflight,
)
from .decoder_det_block_dual_stream import (
    DFRDeterministicDualStreamBlockPort,
    prepare_deterministic_dual_stream_block_port,
    validate_deterministic_dual_stream_block_port,
)
from .decoder_det_dual_stream_exact import (
    DFRExactDeterministicDualStreamPort,
    DFRDeterministicStageExecution,
    prepare_exact_deterministic_dual_stream_port,
    execute_exact_deterministic_stages,
    validate_exact_deterministic_stages,
)
from .decoder_stage5_dual_stream_exact import (
    DFRExactStage5DualStreamPort,
    DFRStage5SingleTileExecution,
    prepare_exact_stage5_dual_stream_port,
    execute_exact_stage5_single_tile,
    validate_exact_stage5_single_tile,
)
from .decoder_full_tiled_exact import (
    DFROfficialTiledDecodeSchedule,
    DFROfficialTiledDecodeExecution,
    prepare_official_auto_tiled_decode_schedule,
    prepare_official_tiled_decode_schedule,
    execute_official_tiled_decode,
    validate_official_tiled_decode,
)
from .decoder_pipeline import decode_spatial_dfr_video, decode_stage1_spatial_dfr_video
from .decoder_bridge import decode_final_video_with_continued_rng, validate_decoder_bridge

from .stage2_output import (
    finalize_stage2_result,
    prepare_official_final_decode,
    split_stage1_audio_video,
    trim_stage1_decoded_audio,
    trim_stage2_result_decoded_audio,
    validate_stage2_final_output,
    validate_stage2_decoded_audio_trim,
)


def _parse_int_csv(text_value: str) -> list[int]:
    raw = str(text_value).strip()
    if not raw:
        raise ValueError("Expected a comma-separated list of integers, got an empty string.")
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected a comma-separated list of integers, got an empty string.")
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"Could not parse integer list from '{text_value}'.") from exc


class LTXDFRResolveDFRCanvas:
    """Exact port of Lightricks ltx_pipelines.dfr_layout.resolve_canvas()."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "requested_frames": (
                    "INT",
                    {
                        "default": 97,
                        "min": 2,
                        "max": 100000,
                        "step": 1,
                        "tooltip": (
                            "Caller-requested pixel-frame count before DFR tail padding. "
                            "Must satisfy (requested_frames-1) % temporal_scale == 0."
                        ),
                    },
                ),
                "temporal_scale": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "LTX VAE temporal scale. LTX-2.5 default is 8.",
                    },
                ),
            },
            "optional": {
                "generated_slot_indices": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "tooltip": (
                            "Optional comma-separated generated DFR slot positions, for example 16,40,88. "
                            "Leave blank for the exact official 24/32-frame grid. Custom positions must be "
                            "strictly increasing, inside the padded canvas, and aligned to temporal_scale."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("LTX_DFR_LAYOUT", "INT")
    RETURN_NAMES = ("dfr_layout", "padded_video_length")
    FUNCTION = "resolve"
    CATEGORY = "LTX/DFR Spatial/Phase B - Layout & Execution"
    DESCRIPTION = (
        "Uses the exact Lightricks resolve_canvas() geometry and returns the pixel-frame length expected by "
        "EmptyLTXVLatentVideo. Leave generated_slot_indices blank for strict official DFR positions; an explicit "
        "list creates a marked custom-slot layout that is propagated consistently through both stages and decoding."
    )

    def resolve(self, requested_frames=97, temporal_scale=8, generated_slot_indices=""):
        custom_text = str(generated_slot_indices or "").strip()
        layout = (
            make_custom_generated_slot_layout(
                requested_frames,
                _parse_int_csv(custom_text),
                temporal_scale,
            )
            if custom_text
            else make_layout(requested_frames, temporal_scale)
        )
        return (layout, int(layout["padded_frames"]))


class LTXExperimentalResolveDenseSlotCanvas16:
    """Experimental 16-frame generated-slot layout; official 24/32 logic remains untouched."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "requested_frames": (
                    "INT",
                    {
                        "default": 97,
                        "min": 2,
                        "max": 100000,
                        "step": 1,
                        "tooltip": (
                            "Caller-requested pixel-frame count. The experimental working canvas "
                            "is padded to a 16-frame generated-slot grid."
                        ),
                    },
                ),
                "temporal_scale": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "LTX VAE temporal scale. It must divide the fixed experimental "
                            "16-frame segment; LTX-2.5 uses 8."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LTX_DFR_LAYOUT", "INT", "INT", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "dfr_layout",
        "padded_frames",
        "segment_length",
        "pixel_frame_indices",
        "padding_frames",
        "padded_latent_frames",
        "report",
    )
    FUNCTION = "resolve"
    CATEGORY = "LTX/DFR Spatial/Experimental"
    DESCRIPTION = (
        "Experimental Spatial-DFR layout with automatically generated keyframe slots every 16 pixel frames. "
        "For 97 frames it produces 16,32,48,64,80,96. It changes only the layout authority; the official "
        "VideoGeneratedKeyframeSlots semantics and all Stage-1/Stage-2 carry-forward logic remain unchanged."
    )

    def resolve(self, requested_frames=97, temporal_scale=8):
        layout = make_experimental_dense_slot_layout(requested_frames, temporal_scale)
        positions_text = ",".join(str(x) for x in layout["pixel_frame_indices"])
        report = (
            f"PASS=True; EXPERIMENTAL=True; layout_kind={layout['layout_kind']}; "
            f"requested_frames={layout['requested_frames']}; padded_frames={layout['padded_frames']}; "
            f"padding_frames={layout['padding_frames']}; segment_length={layout['segment_length']}; "
            f"pixel_frame_indices={positions_text}; generated_slots={len(layout['pixel_frame_indices'])}; "
            f"temporal_scale={layout['temporal_scale']}; "
            f"requested_latent_frames={layout['requested_latent_frames']}; "
            f"padded_latent_frames={layout['padded_latent_frames']}."
        )
        return (
            layout,
            layout["padded_frames"],
            layout["segment_length"],
            positions_text,
            layout["padding_frames"],
            layout["padded_latent_frames"],
            report,
        )


class LTXDFRValidateDFRCanvas:
    """Validate supported layout invariants and optionally check them against a target latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dfr_layout": ("LTX_DFR_LAYOUT",),
            },
            "optional": {
                "target_latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Optional padded working latent. If supplied, its temporal length must match "
                            "the layout's padded canvas exactly."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("passed", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Validates official canvas geometry plus the selected official, dense-test, or custom generated-slot "
        "policy. When target_latent is connected, it also verifies that its pixel-frame span equals padded_frames."
    )

    def validate(self, dfr_layout, target_latent=None):
        layout = validate_layout(dfr_layout)
        latent_match = True
        latent_frames = None
        latent_pixel_frames = None
        if target_latent is not None:
            samples = target_latent.get("samples")
            if samples is None or not torch.is_tensor(samples) or samples.ndim != 5:
                raise ValueError("target_latent must contain [B,C,T,H,W] samples.")
            latent_frames = int(samples.shape[2])
            latent_pixel_frames = (latent_frames - 1) * int(layout["temporal_scale"]) + 1
            latent_match = latent_pixel_frames == int(layout["padded_frames"])

        positions = ",".join(str(x) for x in layout["pixel_frame_indices"])
        passed = bool(latent_match)
        report = (
            f"PASS={passed}; layout_kind={layout.get('layout_kind', 'official')}; "
            f"requested_frames={layout['requested_frames']}; "
            f"padded_frames={layout['padded_frames']}; padding_frames={layout['padding_frames']}; "
            f"segment_length={layout['segment_length']}; pixel_frame_indices={positions}; "
            f"temporal_scale={layout['temporal_scale']}; padded_latent_frames={layout['padded_latent_frames']}; "
            f"target_latent_frames={latent_frames}; target_pixel_frames={latent_pixel_frames}; "
            f"target_matches_padded_canvas={latent_match}."
        )
        return {"ui": {"text": [report]}, "result": (passed, report)}


class LTXDFRSigmaSchedules:
    """Expose the exact official Spatial-DFR Stage-1 and Stage-2 sigma tensors."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("SIGMAS", "SIGMAS", "INT", "INT", "STRING")
    RETURN_NAMES = ("stage_1_sigmas", "stage_2_sigmas", "stage_1_steps", "stage_2_steps", "report")
    FUNCTION = "get_sigmas"
    CATEGORY = "LTX/DFR Spatial/Phase B - Layout & Execution"
    DESCRIPTION = (
        "Outputs the exact Lightricks distilled sigma schedules used by Spatial DFR: "
        "9 sigma points / 8 Euler transitions for Stage 1, and 4 sigma points / 3 transitions for Stage 2. "
        "No generic LTX scheduler or token-dependent sigma shifting is applied."
    )

    def get_sigmas(self):
        s1 = stage_1_sigmas()
        s2 = stage_2_sigmas()
        report = (
            "stage_1_sigmas=" + ",".join(str(v) for v in DISTILLED_SIGMA_VALUES) + "; "
            "stage_1_steps=8; stage_2_sigmas=" + ",".join(str(v) for v in STAGE_2_DISTILLED_SIGMA_VALUES) + "; "
            "stage_2_steps=3; dtype=float32."
        )
        return (s1, s2, int(s1.numel() - 1), int(s2.numel() - 1), report)


class LTXDFRValidateDFRSigmaSchedules:
    """Validate Comfy SIGMAS against the exact upstream DFR constants."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_1_sigmas": ("SIGMAS",),
                "stage_2_sigmas": ("SIGMAS",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "stage_1_max_error", "stage_2_max_error", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    DESCRIPTION = (
        "Requires exact float32 equality with Lightricks DISTILLED_SIGMAS and "
        "STAGE_2_DISTILLED_SIGMAS."
    )
    OUTPUT_NODE = True

    def validate(self, stage_1_sigmas, stage_2_sigmas):
        # Input names shadow the module helper names, so import aliases locally.
        from .dfr_sigmas import stage_1_sigmas as _official_s1, stage_2_sigmas as _official_s2

        expected_s1 = _official_s1()
        expected_s2 = _official_s2()
        pass_s1, err_s1, detail_s1 = validate_sigma_tensor(stage_1_sigmas, expected_s1, "stage_1")
        pass_s2, err_s2, detail_s2 = validate_sigma_tensor(stage_2_sigmas, expected_s2, "stage_2")
        passed = bool(pass_s1 and pass_s2)
        report = f"PASS={passed}; {detail_s1}; {detail_s2}."
        return {"ui": {"text": [report]}, "result": (passed, err_s1, err_s2, report)}


class LTXDFRGaussianNoiser:
    """Exact DFR Gaussian noiser/state materialization, before any model call."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_state_latent": ("LATENT",),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                        "tooltip": "Seed for the official torch.Generator used by GaussianNoiser.",
                    },
                ),
                "noise_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.000001,
                        "tooltip": (
                            "Official GaussianNoiser noise_scale. DFR Stage 1 uses 1.0; "
                            "Stage 2 uses the first Stage-2 sigma (0.909375)."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("noised_official_state", "report")
    FUNCTION = "apply"
    CATEGORY = "LTX/DFR Spatial/Phase B - Execution Bridge"
    DESCRIPTION = (
        "Materializes the strict official token state in BF16 and applies Lightricks GaussianNoiser exactly: "
        "latent=lerp(latent, noise, noise_scale), then lerp(clean_latent, latent, denoise_mask). "
        "This node deliberately performs no transformer/model call."
    )

    def apply(self, official_state_latent, seed=42, noise_scale=1.0):
        out = materialize_official_gaussian_noised_state(
            official_state_latent,
            seed=int(seed),
            noise_scale=float(noise_scale),
        )
        stats = noiser_stats(out)
        report = (
            f"GaussianNoiser materialized; seed={stats['seed']}; noise_scale={stats['noise_scale']:.9g}; "
            f"state_dtype={stats['state_dtype']}; noise_dtype={stats['noise_dtype']}; "
            f"device={stats['device_type']}; token_shape={stats['token_shape']}; "
            f"fully_clean_tokens={stats['fully_clean_tokens']}; fully_noised_tokens={stats['fully_noised_tokens']}; "
            f"partial_tokens={stats['partial_tokens']}; mask_min={stats['mask_min']:.9g}; mask_max={stats['mask_max']:.9g}."
        )
        return (out, report)


class LTXDFRValidateGaussianNoiser:
    """Numerically validate materialized GaussianNoiser output from a pre-noise state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pre_noise_state": ("LATENT",),
                "noised_state": ("LATENT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.000001}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "max_latent_error",
        "max_clean_error",
        "max_mask_error",
        "max_metadata_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    DESCRIPTION = (
        "Recomputes the upstream GaussianNoiser formula independently from the pre-noise state and compares it "
        "against the materialized output. Also verifies BF16 state dtype and preservation of positions/masks/layout."
    )
    OUTPUT_NODE = True

    def validate(self, pre_noise_state, noised_state, seed=42, noise_scale=1.0):
        pre = get_official_state(pre_noise_state)
        post = get_official_state(noised_state)
        pre_tokens = pre.get("token_state")
        post_tokens = post.get("token_state")
        if pre_tokens is None or post_tokens is None:
            raise ValueError("Both states must contain official patchified token_state data.")
        if pre_tokens.get(NOISER_METADATA_KEY) is not None:
            raise ValueError("pre_noise_state is already noised; connect the state immediately before GaussianNoiser.")
        metadata = post_tokens.get(NOISER_METADATA_KEY)
        if metadata is None:
            raise ValueError("noised_state has no GaussianNoiser metadata. Connect GaussianNoiser output.")

        # Independent reconstruction of upstream GaussianNoiser.
        latent0 = pre_tokens["latent"].to(device=post_tokens["latent"].device, dtype=torch.bfloat16)
        clean = pre_tokens["clean_latent"].to(device=post_tokens["latent"].device, dtype=torch.bfloat16)
        mask = pre_tokens["denoise_mask"].to(device=post_tokens["latent"].device, dtype=torch.float32)
        generator = torch.Generator(device=latent0.device).manual_seed(int(seed))
        noise = torch.randn(*latent0.shape, device=latent0.device, dtype=latent0.dtype, generator=generator)
        expected_mixed = torch.lerp(latent0.float(), noise.float(), float(noise_scale))
        expected = torch.lerp(clean.float(), expected_mixed, mask).to(latent0.dtype)

        actual = post_tokens["latent"]
        latent_err = float((actual - expected).abs().max().item()) if actual.numel() else 0.0
        post_clean = post_tokens["clean_latent"]
        clean_err = float((post_clean - clean).abs().max().item()) if post_clean.numel() else 0.0
        post_mask = post_tokens["denoise_mask"]
        mask_err = float((post_mask - mask).abs().max().item()) if post_mask.numel() else 0.0

        positions_preserved = torch.equal(
            post_tokens["positions"],
            pre_tokens["positions"].to(device=post_tokens["positions"].device, dtype=torch.float32),
        )
        keyframes_preserved = True
        if torch.is_tensor(pre_tokens.get("keyframes_mask")):
            keyframes_preserved = torch.equal(
                post_tokens["keyframes_mask"],
                pre_tokens["keyframes_mask"].to(device=post_tokens["keyframes_mask"].device, dtype=torch.float32),
            )
        layout_preserved = post_tokens.get("generated_keyframe_layout") == pre_tokens.get("generated_keyframe_layout")
        attention_preserved = (
            (post_tokens.get("attention_mask") is None and pre_tokens.get("attention_mask") is None)
            or (
                torch.is_tensor(post_tokens.get("attention_mask"))
                and torch.is_tensor(pre_tokens.get("attention_mask"))
                and torch.equal(post_tokens["attention_mask"], pre_tokens["attention_mask"].to(post_tokens["attention_mask"].device))
            )
        )
        dtype_ok = actual.dtype == torch.bfloat16 and post_clean.dtype == torch.bfloat16 and post_mask.dtype == torch.float32
        metadata_ok = (
            int(metadata.get("seed", -1)) == int(seed)
            and abs(float(metadata.get("noise_scale", -999.0)) - float(noise_scale)) <= 1e-12
            and metadata.get("state_dtype") == "bfloat16"
            and metadata.get("noise_dtype") == "bfloat16"
        )
        metadata_error = 0.0 if (positions_preserved and keyframes_preserved and layout_preserved and attention_preserved and dtype_ok and metadata_ok) else 1.0

        # Strong invariants useful for DFR Stage 1 diagnostics.
        clean_mask = post_mask == 0
        full_noise_mask = post_mask == 1
        clean_token_err = 0.0
        if clean_mask.any():
            expanded_clean_mask = clean_mask.expand_as(actual)
            clean_token_err = float((actual[expanded_clean_mask] - post_clean[expanded_clean_mask]).abs().max().item())

        passed = latent_err == 0.0 and clean_err == 0.0 and mask_err == 0.0 and metadata_error == 0.0
        report = (
            f"PASS={passed}; latent_max_abs_error={latent_err:.9g}; clean_max_abs_error={clean_err:.9g}; "
            f"mask_max_abs_error={mask_err:.9g}; metadata_error={metadata_error:.9g}; "
            f"bf16_state={dtype_ok}; positions_preserved={positions_preserved}; keyframes_mask_preserved={keyframes_preserved}; "
            f"generated_layout_preserved={layout_preserved}; attention_mask_preserved={attention_preserved}; "
            f"seed={int(seed)}; noise_scale={float(noise_scale):.9g}; token_shape={tuple(actual.shape)}; "
            f"mask0_clean_token_error={clean_token_err:.9g}; mask0_tokens={int(clean_mask.sum().item())}; "
            f"mask1_tokens={int(full_noise_mask.sum().item())}; operations={len(post.get('operations', []))}."
        )
        return {"ui": {"text": [report]}, "result": (passed, latent_err, clean_err, mask_err, metadata_error, report)}


class LTXDFRCreateStage1AudioState:
    """Create the upstream Stage-1 audio latent state aligned to the official video canvas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_official_state": ("LATENT",),
                "audio_channels": ("INT", {"default": 8, "min": 1, "max": 256, "step": 1}),
                "mel_bins": ("INT", {"default": 16, "min": 1, "max": 1024, "step": 1}),
                "sample_rate": ("INT", {"default": 16000, "min": 1000, "max": 192000, "step": 1}),
                "hop_length": ("INT", {"default": 160, "min": 1, "max": 8192, "step": 1}),
                "audio_latent_downsample_factor": ("INT", {"default": 4, "min": 1, "max": 128, "step": 1}),
                "is_causal": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("audio_official_state", "audio_latent_frames", "duration_seconds", "report")
    FUNCTION = "build"
    CATEGORY = "LTX/DFR Spatial/Phase B - Audio"
    DESCRIPTION = (
        "Creates the official Stage-1 audio latent state used by LTXAV: zero clean/current audio latents, "
        "all-1 denoise mask, and the official AudioPatchifier timing positions derived from the connected "
        "video official state's padded canvas duration."
    )

    def build(
        self,
        video_official_state,
        audio_channels=8,
        mel_bins=16,
        sample_rate=16000,
        hop_length=160,
        audio_latent_downsample_factor=4,
        is_causal=True,
    ):
        out = create_audio_official_state_from_video(
            video_official_state,
            channels=int(audio_channels),
            mel_bins=int(mel_bins),
            sample_rate=int(sample_rate),
            hop_length=int(hop_length),
            audio_latent_downsample_factor=int(audio_latent_downsample_factor),
            is_causal=bool(is_causal),
        )
        stats = audio_state_stats(out)
        report = (
            f"audio_shape={stats['audio_shape']}; token_shape={stats['token_shape']}; duration_seconds={stats['duration_seconds']:.9g}; "
            f"pixel_frames={stats['pixel_frames']}; fps={stats['fps']:.9g}; sample_rate={stats['sample_rate']}; "
            f"hop_length={stats['hop_length']}; audio_latent_downsample_factor={stats['audio_latent_downsample_factor']}; "
            f"first_timing=[{stats['pos0_start']:.9g}, {stats['pos0_end']:.9g}]; "
            f"last_timing=[{stats['pos_last_start']:.9g}, {stats['pos_last_end']:.9g}]."
        )
        return (out, int(stats['audio_shape'][2]), float(stats['duration_seconds']), report)


class LTXDFRValidateStage1AudioState:
    """Validate that the created audio state exactly matches the connected video canvas timing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_official_state": ("LATENT",),
                "audio_official_state": ("LATENT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "shape_error",
        "position_error",
        "duration_error",
        "token_shape_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    DESCRIPTION = (
        "Checks that the audio official state uses the exact latent-frame count, token shape, duration, and "
        "AudioPatchifier timing positions implied by the connected official video state."
    )
    OUTPUT_NODE = True

    def validate(self, video_official_state, audio_official_state):
        result = validate_audio_state_against_video(audio_official_state, video_official_state)
        report = (
            f"PASS={result['passed']}; audio_shape={result['audio_shape']}; expected_shape={result['expected_shape']}; "
            f"token_shape={result['token_shape']}; expected_token_shape={result['expected_token_shape']}; "
            f"pixel_frames={result['pixel_frames']}; fps={result['fps']:.9g}; expected_duration={result['expected_duration']:.9g}; "
            f"shape_error={result['shape_error']:.9g}; token_shape_error={result['token_shape_error']:.9g}; "
            f"mask_shape_error={result['mask_shape_error']:.9g}; position_error={result['position_error']:.9g}; "
            f"duration_error={result['duration_error']:.9g}; all_one_mask={result['all_one_mask']}."
        )
        return {"ui": {"text": [report]}, "result": (
            result['passed'],
            result['shape_error'],
            result['position_error'],
            result['duration_error'],
            result['token_shape_error'],
            report,
        )}


class LTXDFRAudioGaussianNoiser:
    """Materialize the official Gaussian noiser for the Stage-1 audio state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_official_state": ("LATENT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.000001}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("noised_audio_state", "report")
    FUNCTION = "materialize"
    CATEGORY = "LTX/DFR Spatial/Phase B - Audio"
    DESCRIPTION = (
        "Applies the exact upstream two-lerp Gaussian noiser formula to the official Stage-1 audio token state, "
        "materializing BF16 noised audio tokens plus metadata for later joint LTXAV execution."
    )

    def materialize(self, audio_official_state, seed=42, noise_scale=1.0):
        out = materialize_audio_gaussian_noised_state(
            audio_official_state,
            seed=int(seed),
            noise_scale=float(noise_scale),
        )
        stats = audio_state_stats(out)
        report = (
            f"Audio GaussianNoiser materialized; seed={int(seed)}; noise_scale={float(noise_scale):.9g}; "
            f"audio_shape={stats['audio_shape']}; token_shape={stats['token_shape']}; duration_seconds={stats['duration_seconds']:.9g}; "
            f"pixel_frames={stats['pixel_frames']}; fps={stats['fps']:.9g}."
        )
        return (out, report)


class LTXDFRValidateAudioGaussianNoiser:
    """Numerically validate the materialized audio Gaussian noiser against the pre-noise state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pre_noise_audio_state": ("LATENT",),
                "noised_audio_state": ("LATENT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.000001}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "max_latent_error",
        "max_clean_error",
        "max_mask_error",
        "max_metadata_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    DESCRIPTION = (
        "Recomputes the official audio Gaussian noiser independently from the pre-noise state and compares it "
        "with the materialized output, including metadata and preserved timing positions."
    )
    OUTPUT_NODE = True

    def validate(self, pre_noise_audio_state, noised_audio_state, seed=42, noise_scale=1.0):
        result = validate_audio_gaussian_noiser(
            pre_noise_audio_state,
            noised_audio_state,
            seed=int(seed),
            noise_scale=float(noise_scale),
        )
        report = (
            f"PASS={result['passed']}; latent_max_abs_error={result['latent_error']:.9g}; "
            f"clean_max_abs_error={result['clean_error']:.9g}; mask_max_abs_error={result['mask_error']:.9g}; "
            f"position_error={result['position_error']:.9g}; metadata_error={result['metadata_error']:.9g}; "
            f"token_shape={result['token_shape']}; dtype_ok={result['dtype_ok']}; "
            f"metadata_ok={result['metadata_ok']}; structure_ok={result['structure_ok']}; "
            f"noise_scale={float(noise_scale):.9g}; seed={int(seed)}."
        )
        return {"ui": {"text": [report]}, "result": (
            result['passed'],
            result['latent_error'],
            result['clean_error'],
            result['mask_error'],
            result['metadata_error'],
            report,
        )}



class LTXDFRStage1AVGaussianNoiser:
    """Noise Stage-1 video then audio from one shared generator, matching upstream DFR."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_official_state": ("LATENT",),
                "audio_official_state": ("LATENT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.000001}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("noised_video_state", "noised_audio_state", "report")
    FUNCTION = "materialize"
    CATEGORY = "LTX/DFR Spatial/Phase B - Audio"
    DESCRIPTION = (
        "Official Stage-1 AV noiser. Creates one seeded torch.Generator on Comfy's active execution device, "
        "samples VIDEO noise first and AUDIO noise second from the same advanced RNG stream, then applies the "
        "official BF16 GaussianNoiser formula to both modalities. Use this node for the final parity path."
    )

    def materialize(self, video_official_state, audio_official_state, seed=42, noise_scale=1.0):
        video, audio, stats = materialize_stage1_av_gaussian_noised_states(
            video_official_state,
            audio_official_state,
            seed=int(seed),
            noise_scale=float(noise_scale),
        )
        report = (
            f"Stage1 AV GaussianNoiser; seed={stats['seed']}; noise_scale={stats['noise_scale']:.9g}; "
            f"device={stats['device']}; rng_order={stats['rng_order']}; rng_state_bytes={stats['rng_state_bytes']}; "
            f"video_token_shape={stats['video_token_shape']}; audio_token_shape={stats['audio_token_shape']}."
        )
        return (video, audio, report)


class LTXDFRValidateStage1AVGaussianNoiser:
    """Validate exact shared-generator video->audio RNG sequencing."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pre_noise_video_state": ("LATENT",),
                "pre_noise_audio_state": ("LATENT",),
                "noised_video_state": ("LATENT",),
                "noised_audio_state": ("LATENT",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "noise_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.000001}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "video_max_error", "audio_max_error", "metadata_error", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Independently recreates one generator seeded once, consumes the video draw first and audio draw second, "
        "and requires exact equality with both outputs of the Stage-1 AV Gaussian Noiser."
    )

    def validate(self, pre_noise_video_state, pre_noise_audio_state, noised_video_state, noised_audio_state, seed=42, noise_scale=1.0):
        r = validate_stage1_av_gaussian_noiser(
            pre_noise_video_state,
            pre_noise_audio_state,
            noised_video_state,
            noised_audio_state,
            seed=int(seed),
            noise_scale=float(noise_scale),
        )
        report = (
            f"PASS={r['passed']}; video_max_abs_error={r['video_error']:.9g}; "
            f"audio_max_abs_error={r['audio_error']:.9g}; video_clean_error={r['video_clean_error']:.9g}; "
            f"audio_clean_error={r['audio_clean_error']:.9g}; video_mask_error={r['video_mask_error']:.9g}; "
            f"audio_mask_error={r['audio_mask_error']:.9g}; metadata_error={r['metadata_error']:.9g}; "
            f"metadata_ok={r['metadata_ok']}; rng_state_ok={r['rng_state_ok']}; rng_state_bytes={r['rng_state_bytes']}; "
            f"device={r['device']}; rng_order=video_then_audio; "
            f"video_token_shape={r['video_token_shape']}; audio_token_shape={r['audio_token_shape']}."
        )
        return {"ui": {"text": [report]}, "result": (r['passed'], r['video_error'], r['audio_error'], r['metadata_error'], report)}


class LTXDFRMaterializeStage1AVModelInput:
    """Build the joint Stage-1 AV transformer-input description without running the model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_video_state": ("LATENT",),
                "noised_audio_state": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_AV_MODEL_INPUT", "INT", "INT", "STRING")
    RETURN_NAMES = ("av_model_input", "video_token_count", "audio_token_count", "report")
    FUNCTION = "materialize"
    CATEGORY = "LTX/DFR Spatial/Phase B - Audio"
    DESCRIPTION = (
        "Sub-step B1 of the joint LTXAV bridge. Verifies the connected noised official VIDEO and AUDIO states, "
        "then packs them into one deterministic Stage-1 AV model-input description with video tokens first and "
        "audio tokens second. It does not yet invoke the transformer."
    )

    def materialize(self, noised_video_state, noised_audio_state):
        model_input = materialize_stage1_av_model_input(noised_video_state, noised_audio_state)
        stats = av_model_input_stats(model_input)
        report = (
            f"bridge_version={stats['version']}; total_token_count={stats['total_token_count']}; "
            f"joint_mask_shape={stats['joint_mask_shape']}; joint_modality_ids_shape={stats['joint_modality_ids_shape']}; "
            f"video_token_shape={stats['video_token_shape']}; audio_token_shape={stats['audio_token_shape']}; "
            f"video_positions_shape={stats['video_positions_shape']}; audio_positions_shape={stats['audio_positions_shape']}; "
            f"video_keyframes_mask_shape={stats['video_keyframes_mask_shape']}; video_feature_dim={stats['video_feature_dim']}; "
            f"audio_feature_dim={stats['audio_feature_dim']}; video_token_range={stats['video_token_range']}; "
            f"audio_token_range={stats['audio_token_range']}; frame_rate={stats['frame_rate']:.9g}; "
            f"duration_seconds={stats['duration_seconds']:.9g}; device={stats['device']}; "
            f"video_dtype={stats['video_dtype']}; audio_dtype={stats['audio_dtype']}; modality_order={stats['modality_order']}."
        )
        return (model_input, int(model_input.video_token_count), int(model_input.audio_token_count), report)


class LTXDFRValidateStage1AVModelInput:
    """Validate the joint Stage-1 AV input builder before the real transformer step."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_video_state": ("LATENT",),
                "noised_audio_state": ("LATENT",),
                "av_model_input": ("LTX_DFR_AV_MODEL_INPUT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "joint_mask_error",
        "video_position_error",
        "audio_position_error",
        "metadata_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Requires the AV model-input object to reproduce, exactly, the connected noised VIDEO/AUDIO token states, "
        "their masks, their modality-specific positions, and the advertised video-first / audio-second token ranges."
    )

    def validate(self, noised_video_state, noised_audio_state, av_model_input):
        if not isinstance(av_model_input, DFRStage1AVModelInput):
            raise ValueError(f"av_model_input has unexpected type {type(av_model_input)}.")
        r = validate_stage1_av_model_input(noised_video_state, noised_audio_state, av_model_input)
        report = (
            f"PASS={r['passed']}; video_max_abs_error={r['video_error']:.9g}; "
            f"audio_max_abs_error={r['audio_error']:.9g}; video_mask_error={r['video_mask_error']:.9g}; "
            f"audio_mask_error={r['audio_mask_error']:.9g}; joint_mask_error={r['joint_mask_error']:.9g}; "
            f"modality_ids_error={r['modality_ids_error']:.9g}; video_position_error={r['video_position_error']:.9g}; "
            f"audio_position_error={r['audio_position_error']:.9g}; keyframes_mask_error={r['keyframes_mask_error']:.9g}; "
            f"ordering_error={r['ordering_error']:.9g}; metadata_error={r['metadata_error']:.9g}; "
            f"total_token_count={r['total_token_count']}; joint_mask_shape={r['joint_mask_shape']}; "
            f"joint_modality_ids_shape={r['joint_modality_ids_shape']}; video_token_shape={r['video_token_shape']}; "
            f"audio_token_shape={r['audio_token_shape']}; video_positions_shape={r['video_positions_shape']}; "
            f"audio_positions_shape={r['audio_positions_shape']}; video_token_range={r['video_token_range']}; "
            f"audio_token_range={r['audio_token_range']}; frame_rate={r['frame_rate']:.9g}; "
            f"duration_seconds={r['duration_seconds']:.9g}; device={r['device']}; "
            f"audio_alignment_passed={r['audio_alignment_passed']}."
        )
        return {"ui": {"text": [report]}, "result": (
            r['passed'],
            r['joint_mask_error'],
            r['video_position_error'],
            r['audio_position_error'],
            r['metadata_error'],
            report,
        )}



class LTXDFRExecuteStage1AVDenoise:
    """Run one real joint Stage-1 AV transformer/X0 evaluation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "noised_video_state": ("LATENT",),
                "noised_audio_state": ("LATENT",),
                "sigmas": ("SIGMAS",),
                "step_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": (
                            "Official distilled Spatial-DFR Stage 1 uses CFG=1.0. Keep this at 1.0 for parity."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("raw_denoised_video_state", "raw_denoised_audio_state", "report")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/Phase B - Audio"
    DESCRIPTION = (
        "Sub-step B2 of the joint LTXAV parity bridge. Rebuilds the official Stage-1 AV model input, "
        "materializes the official video/audio state through Comfy's native LTXAV bridge and runs one real joint "
        "transformer/X0 evaluation. The execution seed is recovered automatically from the upstream shared AV "
        "Gaussian-noiser metadata; there is no independent B2 seed. Returns strict official VIDEO and AUDIO states "
        "whose token latent is the raw x0 prediction for the current sigma step."
    )

    def run(self, model, positive, negative, noised_video_state, noised_audio_state, sigmas, step_index=0, cfg_scale=1.0):
        denoised_video, denoised_audio, _av_input, report = execute_stage1_av_denoise(
            model=model,
            positive=positive,
            negative=negative,
            noised_video_state=noised_video_state,
            noised_audio_state=noised_audio_state,
            sigmas=sigmas,
            step_index=step_index,
            cfg_scale=cfg_scale,
            seed=None,
        )
        return (denoised_video, denoised_audio, report)


class LTXDFRValidateStage1AVDenoise:
    """Structural sanity-check for the first real Stage-1 AV denoise pass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_video_state": ("LATENT",),
                "noised_audio_state": ("LATENT",),
                "raw_denoised_video_state": ("LATENT",),
                "raw_denoised_audio_state": ("LATENT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "video_shape_error",
        "audio_shape_error",
        "video_preview_error",
        "audio_preview_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Checks that the Stage-1 AV denoise outputs preserve the strict official token geometry, keep both "
        "preview LATENT tensors exactly synchronized with their token-state views, and contain only finite values."
    )

    def validate(self, noised_video_state, noised_audio_state, raw_denoised_video_state, raw_denoised_audio_state):
        r = validate_stage1_av_denoise(
            noised_video_state,
            noised_audio_state,
            raw_denoised_video_state,
            raw_denoised_audio_state,
        )
        report = (
            f"PASS={r['passed']}; video_shape_error={r['video_shape_error']:.9g}; "
            f"audio_shape_error={r['audio_shape_error']:.9g}; video_preview_error={r['video_preview_error']:.9g}; "
            f"audio_preview_error={r['audio_preview_error']:.9g}; video_finite_error={r['video_finite_error']:.9g}; "
            f"audio_finite_error={r['audio_finite_error']:.9g}; video_delta={r['video_delta']:.9g}; "
            f"audio_delta={r['audio_delta']:.9g}; video_token_shape={r['video_token_shape']}; "
            f"audio_token_shape={r['audio_token_shape']}; video_device={r['video_device']}; audio_device={r['audio_device']}."
        )
        return {"ui": {"text": [report]}, "result": (
            r['passed'],
            r['video_shape_error'],
            r['audio_shape_error'],
            r['video_preview_error'],
            r['audio_preview_error'],
            report,
        )}


class LTXDFRStage1AVEulerStepFromDenoised:
    """Apply exact official post-process + deterministic Euler to both Stage-1 modalities."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_video_state": ("LATENT",),
                "noised_audio_state": ("LATENT",),
                "raw_denoised_video_state": ("LATENT",),
                "raw_denoised_audio_state": ("LATENT",),
                "sigmas": ("SIGMAS",),
                "step_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT", "STRING")
    RETURN_NAMES = (
        "next_video_state",
        "next_audio_state",
        "post_processed_video_state",
        "post_processed_audio_state",
        "report",
    )
    FUNCTION = "step"
    CATEGORY = "LTX/DFR Spatial/Phase B - Audio"
    DESCRIPTION = (
        "Sub-step B3. Mirrors the official LTX joint euler_denoising_loop after one AV transformer evaluation: "
        "post_process_latent is applied independently to VIDEO and AUDIO using each modality's clean_latent and "
        "denoise_mask, then EulerDiffusionStep advances both from sigma[step_index] to sigma[step_index+1]."
    )

    def step(
        self,
        noised_video_state,
        noised_audio_state,
        raw_denoised_video_state,
        raw_denoised_audio_state,
        sigmas,
        step_index=0,
    ):
        return stage1_av_euler_step_from_denoised(
            noised_video_state,
            noised_audio_state,
            raw_denoised_video_state,
            raw_denoised_audio_state,
            sigmas,
            step_index,
        )


class LTXDFRValidateStage1AVEulerStep:
    """Exact-value validator for the joint AV post-process + Euler transition."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_video_state": ("LATENT",),
                "noised_audio_state": ("LATENT",),
                "raw_denoised_video_state": ("LATENT",),
                "raw_denoised_audio_state": ("LATENT",),
                "next_video_state": ("LATENT",),
                "next_audio_state": ("LATENT",),
                "post_processed_video_state": ("LATENT",),
                "post_processed_audio_state": ("LATENT",),
                "sigmas": ("SIGMAS",),
                "step_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "video_next_error",
        "audio_next_error",
        "video_post_error",
        "audio_post_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Independently recomputes the exact VIDEO and AUDIO post-process/Euler transition and checks all latent "
        "values plus preservation of clean_latent, denoise_mask and positions."
    )

    def validate(
        self,
        noised_video_state,
        noised_audio_state,
        raw_denoised_video_state,
        raw_denoised_audio_state,
        next_video_state,
        next_audio_state,
        post_processed_video_state,
        post_processed_audio_state,
        sigmas,
        step_index=0,
    ):
        r = validate_stage1_av_euler_step(
            noised_video_state,
            noised_audio_state,
            raw_denoised_video_state,
            raw_denoised_audio_state,
            next_video_state,
            next_audio_state,
            post_processed_video_state,
            post_processed_audio_state,
            sigmas,
            step_index,
        )
        report = (
            f"PASS={r['passed']}; sigma={r['sigma']:.9g}; sigma_next={r['sigma_next']:.9g}; "
            f"video_next_error={r['video_next_error']:.9g}; audio_next_error={r['audio_next_error']:.9g}; "
            f"video_post_error={r['video_post_error']:.9g}; audio_post_error={r['audio_post_error']:.9g}; "
            f"video_mask_error={r['video_mask_error']:.9g}; audio_mask_error={r['audio_mask_error']:.9g}; "
            f"video_position_error={r['video_position_error']:.9g}; audio_position_error={r['audio_position_error']:.9g}; "
            f"video_clean_error={r['video_clean_error']:.9g}; audio_clean_error={r['audio_clean_error']:.9g}; "
            f"video_token_shape={r['video_token_shape']}; audio_token_shape={r['audio_token_shape']}."
        )
        return {"ui": {"text": [report]}, "result": (
            r['passed'],
            r['video_next_error'],
            r['audio_next_error'],
            r['video_post_error'],
            r['audio_post_error'],
            report,
        )}


class LTXDFRValidateDFRModelCapability:
    """Fail fast unless the connected transformer supports DFR generated-keyframe slots."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "BOOLEAN", "STRING")
    RETURN_NAMES = ("model", "supported", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Preflight"
    DESCRIPTION = (
        "Current DFR requires a transformer trained with use_keyframes_abs_pos_embedding=True. "
        "Put this directly after the MODEL loader and use its MODEL output for Stage 1 and Stage 2."
    )

    def validate(self, model):
        evidence = assert_generated_keyframes_supported(model)
        report = f"PASS=True; generated_keyframes_supported=True; evidence={evidence}."
        return {"ui": {"text": [report]}, "result": (model, True, report)}


class LTXDFRRunStage1AVLoop:
    """Run the complete official deterministic Stage-1 AV trajectory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "noised_video_state": ("LATENT",),
                "noised_audio_state": ("LATENT",),
                "stage_1_sigmas": ("SIGMAS",),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Keep 1.0 for the distilled official-parity Stage-1 AV test.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("final_video_state", "final_audio_state", "stage_1_base_latent", "generated_keyframes", "report")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/Phase B - Audio"
    DESCRIPTION = (
        "Runs the complete official Stage-1 AV trajectory by iterating across the exact 8-transition distilled "
        "sigma schedule, executing one real joint LTXAV denoise at each step, then applying the validated joint "
        "post-process + Euler advance. Returns the final strict official VIDEO and AUDIO states together with the "
        "exported Stage-1 base video latent and generated keyframe stack."
    )

    def run(self, model, positive, negative, noised_video_state, noised_audio_state, stage_1_sigmas, cfg_scale=1.0):
        return execute_stage1_av_denoising_loop(
            model=model,
            positive=positive,
            negative=negative,
            noised_video_state=noised_video_state,
            noised_audio_state=noised_audio_state,
            sigmas=stage_1_sigmas,
            cfg_scale=cfg_scale,
        )


class LTXDFRRunStage1SpatialDFR:
    """Run the complete public Stage-1 path and return one frozen handoff."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "video_official_state": (
                    "LATENT",
                    {"tooltip": "Conditioned pre-noise video state with generated Stage-1 keyframe slots."},
                ),
                "audio_official_state": (
                    "LATENT",
                    {"tooltip": "Pre-noise official Stage-1 audio state aligned to the video."},
                ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Keep 1.0 for strict official Spatial-DFR Stage-1 parity.",
                    },
                ),
            },
            "optional": {
                "negative": (
                    "CONDITIONING",
                    {"tooltip": "Optional Comfy CFG extension. Strict official parity does not require it."},
                ),
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Optional advanced schedule override. When disconnected, the exact official Stage-1 "
                            "distilled schedule is used."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("LTX_DFR_STAGE2_HANDOFF",)
    RETURN_NAMES = ("stage_1_handoff",)
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/Stage 1"
    DESCRIPTION = (
        "Runs the required Stage-1 operations as one unit: shared VIDEO->AUDIO Gaussian noise, the optimized "
        "whole-schedule AV denoising loop, base/keyframe extraction, and frozen RNG/state handoff. The single "
        "stage_1_handoff output contains the reserved half-resolution video, Stage-1 audio, generated keyframes, "
        "and continued RNG metadata."
    )

    def run(
        self,
        model,
        positive,
        video_official_state,
        audio_official_state,
        seed=42,
        cfg_scale=1.0,
        negative=None,
        sigmas=None,
    ):
        return (
            run_stage1_spatial_dfr(
                model=model,
                positive=positive,
                negative=negative,
                video_official_state=video_official_state,
                audio_official_state=audio_official_state,
                seed=int(seed),
                sigmas=sigmas,
                cfg_scale=float(cfg_scale),
            ),
        )


class LTXDFRValidateStage1AVLoop:
    """Validate the exported outputs of the full Stage-1 AV loop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_video_state": ("LATENT",),
                "final_audio_state": ("LATENT",),
                "stage_1_base_latent": ("LATENT",),
                "generated_keyframes": ("LATENT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "base_max_error",
        "generated_keyframes_max_error",
        "audio_preview_error",
        "metadata_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Checks that the full Stage-1 AV loop exported base/video keyframe outputs as exact slices of the final "
        "strict official video state, and that the final audio state still round-trips exactly from its token view."
    )

    def validate(self, final_video_state, final_audio_state, stage_1_base_latent, generated_keyframes):
        r = validate_stage1_av_loop_outputs(
            final_video_state,
            final_audio_state,
            stage_1_base_latent,
            generated_keyframes,
        )
        report = (
            f"PASS={r['passed']}; base_max_error={r['base_max_abs_error']:.9g}; "
            f"generated_keyframes_max_error={r['generated_keyframes_max_abs_error']:.9g}; "
            f"audio_preview_error={r['audio_preview_error']:.9g}; metadata_error={r['metadata_error']:.9g}; "
            f"shared_seed={r['shared_seed']}; video_base_shape={r['video_base_shape']}; "
            f"generated_keyframes_shape={r['generated_keyframes_shape']}; audio_shape={r['audio_shape']}; "
            f"audio_token_shape={r['audio_token_shape']}; duration_seconds={r['duration_seconds']:.9g}."
        )
        return {"ui": {"text": [report]}, "result": (
            r['passed'],
            r['base_max_abs_error'],
            r['generated_keyframes_max_abs_error'],
            r['audio_preview_error'],
            r['metadata_error'],
            report,
        )}


class LTXDFRPrepareStage2Handoff:
    """Freeze the exact Stage-1 -> Stage-2 parity boundary."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_video_state": ("LATENT",),
                "final_audio_state": ("LATENT",),
                "stage_1_base_latent": ("LATENT",),
                "generated_keyframes": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_STAGE2_HANDOFF", "LATENT", "LATENT", "LATENT", "STRING")
    RETURN_NAMES = (
        "stage_2_handoff",
        "reserved_half_res_video",
        "stage_1_audio_latent",
        "stage_1_generated_keyframes",
        "report",
    )
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - Handoff"
    DESCRIPTION = (
        "Stage 2A. Freezes the exact upstream Stage-1 outputs needed by Stage 2: the reserved half-resolution "
        "base video, generated DFR keyframe stack, Stage-1 audio latent, and the continued Gaussian-noiser RNG "
        "state after Stage-1 VIDEO then AUDIO noise draws. This node performs no upsampling and no Stage-2 noising."
    )

    def prepare(self, final_video_state, final_audio_state, stage_1_base_latent, generated_keyframes):
        return prepare_stage2_handoff(
            final_video_state,
            final_audio_state,
            stage_1_base_latent,
            generated_keyframes,
        )


class LTXDFRValidateStage2Handoff:
    """Validate exact Stage-2A outputs and RNG continuation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": ("LTX_DFR_STAGE2_HANDOFF",),
                "final_video_state": ("LATENT",),
                "final_audio_state": ("LATENT",),
                "reserved_half_res_video": ("LATENT",),
                "stage_1_audio_latent": ("LATENT",),
                "stage_1_generated_keyframes": ("LATENT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "video_error",
        "keyframes_error",
        "audio_error",
        "rng_state_error",
        "next_rng_draw_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Independently checks that Stage 2A preserved the exact Stage-1 base video, keyframes and audio latent, "
        "then replays the Stage-1 shared RNG stream from the seed and verifies both the captured generator state "
        "and the next random draw. All reported numeric errors must be zero."
    )

    def validate(
        self,
        stage_2_handoff,
        final_video_state,
        final_audio_state,
        reserved_half_res_video,
        stage_1_audio_latent,
        stage_1_generated_keyframes,
    ):
        if not isinstance(stage_2_handoff, DFRStage2Handoff):
            raise ValueError(f"stage_2_handoff has unexpected type {type(stage_2_handoff).__name__}.")
        r = validate_stage2_handoff(
            stage_2_handoff,
            final_video_state,
            final_audio_state,
            reserved_half_res_video,
            stage_1_audio_latent,
            stage_1_generated_keyframes,
        )
        report = (
            f"PASS={r['passed']}; video_error={r['video_error']:.9g}; keyframes_error={r['keyframes_error']:.9g}; "
            f"audio_error={r['audio_error']:.9g}; handoff_video_error={r['handoff_video_error']:.9g}; "
            f"handoff_keyframes_error={r['handoff_keyframes_error']:.9g}; handoff_audio_error={r['handoff_audio_error']:.9g}; "
            f"rng_state_error={r['rng_state_error']:.9g}; next_rng_draw_error={r['next_rng_draw_error']:.9g}; "
            f"seed={r['seed']}; rng_state_bytes={r['rng_state_bytes']}; device={r['device']}; "
            f"reserved_video_shape={r['reserved_video_shape']}; keyframes_shape={r['keyframes_shape']}; "
            f"audio_shape={r['audio_shape']}."
        )
        return {"ui": {"text": [report]}, "result": (
            r['passed'],
            r['video_error'],
            r['keyframes_error'],
            r['audio_error'],
            r['rng_state_error'],
            r['next_rng_draw_error'],
            report,
        )}


class LTXDFRRunStage2SpatialUpscale:
    """Run the native x2 latent upscaler on Stage-1 base video + generated keyframes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_1_handoff": ("LTX_DFR_STAGE2_HANDOFF",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_UPSCALED_STAGE1_HANDOFF", "LATENT")
    RETURN_NAMES = ("upscaled_stage_1_handoff", "upscaled_video_latent")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - Upscale"
    DESCRIPTION = (
        "Stage 2B. Runs the native LTXVLatentUpsampler exactly where upstream DFR does: once on the "
        "reserved Stage-1 half-resolution base video and once on the generated keyframe-slot stack, "
        "both read directly from the consolidated Stage-1 handoff. The enriched handoff carries the "
        "upscaled slot seeds plus the preserved Stage-1 audio/RNG internally; the video latent remains "
        "a separate output because user image conditions must be reapplied to it at full resolution."
    )

    def run(self, stage_1_handoff, upscale_model, vae):
        return prepare_stage2_spatial_upscale_from_handoff(
            stage_1_handoff,
            upscale_model,
            vae,
        )


class LTXDFRRunModularStage2SpatialUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source_handoff": ("LTX_DFR_STAGE2_HANDOFF,LTX_DFR_STAGE2_RESULT_HANDOFF,LTX_DFR_TEMPORAL_HANDOFF",),
            "upscale_model": ("LATENT_UPSCALE_MODEL",), "vae": ("VAE",),
        }, "optional": {
            "dfr_layout": ("LTX_DFR_LAYOUT", {"tooltip":"Only needed for older Stage 1/Stage 2 handoffs without embedded layout. Current handoffs carry it automatically; temporal sources derive it internally."}),
        }}
    RETURN_TYPES = ("LTX_DFR_UPSCALED_STAGE1_HANDOFF", "LATENT", "LTX_DFR_LAYOUT", "FLOAT")
    RETURN_NAMES = ("upscaled_stage_1_handoff", "upscaled_video_latent", "dfr_layout", "fps")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/03 - Stage 2"
    DESCRIPTION = "Spatial x2 from Stage 1, completed Stage 2, or a completed temporal round. Feed the outputs to the existing Stage 2 conditioning and AV loop. Use the returned layout for conditioning, output and decode. Reapply user images at the new resolution. Preserves source audio, FPS, duration and RNG continuation."
    def run(self, source_handoff, upscale_model, vae, dfr_layout=None):
        from .modular_stage2 import run_modular_spatial_upscale
        return run_modular_spatial_upscale(source_handoff, upscale_model, vae, dfr_layout)


class LTXDFRRunStage2SpatialUpscaleFromStage2HandoffTest:
    """TEST ONLY: run another native x2 spatial pass from a completed Stage-2 handoff."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": ("LTX_DFR_STAGE2_RESULT_HANDOFF",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_UPSCALED_STAGE1_HANDOFF", "LATENT")
    RETURN_NAMES = ("upscaled_stage_1_handoff", "upscaled_video_latent")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/03 - Stage 2"
    DESCRIPTION = (
        "TEST PURPOSE ONLY. Accepts the completed stage_2_handoff from 'Run Stage 2 AV Loop', "
        "adapts its Stage-2 video, generated keyframes and continued RNG state into the existing "
        "Stage-1-shaped handoff, then runs the exact same native x2 latent upscaler used by the normal "
        "Stage-2 spatial path. Stage-1 audio is intentionally preserved. This is an experimental wiring "
        "shortcut for testing a third spatial refinement stage, not a clean generalized multi-stage API."
    )

    def run(self, stage_2_handoff, upscale_model, vae):
        return prepare_stage2_spatial_upscale_from_stage2_result_for_test(
            stage_2_handoff,
            upscale_model,
            vae,
        )


class LTXDFRValidateStage2SpatialUpscale:
    """Validate Stage-2B native latent upscaling and audio passthrough."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": ("LTX_DFR_STAGE2_HANDOFF",),
                "reserved_half_res_video": ("LATENT",),
                "stage_1_audio_latent": ("LATENT",),
                "stage_1_generated_keyframes": ("LATENT",),
                "upscaled_video_latent": ("LATENT",),
                "stage_1_audio_for_stage2": ("LATENT",),
                "upscaled_generated_keyframes": ("LATENT",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "input_video_error",
        "input_keyframes_error",
        "input_audio_error",
        "video_error",
        "keyframes_error",
        "audio_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Independently re-runs the native LTXVLatentUpsampler on the Stage-2A handoff tensors, "
        "verifies exact x2 output geometry for video and generated keyframes, and checks that "
        "Stage-1 audio was carried forward unchanged."
    )

    def validate(
        self,
        stage_2_handoff,
        reserved_half_res_video,
        stage_1_audio_latent,
        stage_1_generated_keyframes,
        upscaled_video_latent,
        stage_1_audio_for_stage2,
        upscaled_generated_keyframes,
        upscale_model,
        vae,
    ):
        r = validate_stage2_spatial_upscale(
            stage_2_handoff,
            reserved_half_res_video,
            stage_1_audio_latent,
            stage_1_generated_keyframes,
            upscaled_video_latent,
            stage_1_audio_for_stage2,
            upscaled_generated_keyframes,
            upscale_model,
            vae,
        )
        report = (
            f"PASS={r['passed']}; input_video_error={r['input_video_error']:.9g}; "
            f"input_keyframes_error={r['input_keyframes_error']:.9g}; "
            f"input_audio_error={r['input_audio_error']:.9g}; video_error={r['video_error']:.9g}; "
            f"keyframes_error={r['keyframes_error']:.9g}; audio_error={r['audio_error']:.9g}; "
            f"video_shape_error={r['video_shape_error']:.9g}; "
            f"keyframes_shape_error={r['keyframes_shape_error']:.9g}; scale_factor={r['scale_factor']}; "
            f"seed={r['seed']}; native_node={r['native_node']}; "
            f"reserved_video_shape={r['reserved_video_shape']}; upscaled_video_shape={r['upscaled_video_shape']}; "
            f"keyframes_shape={r['keyframes_shape']}; upscaled_keyframes_shape={r['upscaled_keyframes_shape']}; "
            f"audio_shape={r['audio_shape']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                r['passed'],
                r['input_video_error'],
                r['input_keyframes_error'],
                r['input_audio_error'],
                r['video_error'],
                r['keyframes_error'],
                r['audio_error'],
                report,
            ),
        }


class LTXDFRResolveStage2DetailingMetadata:
    """Resolve the selected detailing LoRA metadata before Stage 2C is assembled."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (
                    lora_choices(),
                    {
                        "default": OFFICIAL_DETAILING_LORA_BASENAME,
                        "tooltip": (
                            "Select the official detailing IC-LoRA once here. Its safetensors metadata is read "
                            "before Stage 2C so reference_downscale_factor is not hardcoded downstream."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LTX_DFR_DETAILING_SPEC", "STRING")
    RETURN_NAMES = ("detailing_spec", "report")
    FUNCTION = "resolve"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    DESCRIPTION = (
        "U1.2 preflight. Reads reference_downscale_factor directly from the selected IC-LoRA safetensors metadata. "
        "Feed detailing_spec to both Stage 2C and Stage 2D."
    )

    def resolve(self, lora_name):
        spec, _factor, report = resolve_stage2_detailing_spec(lora_name)
        return (spec, report)


class LTXDFRPrepareStage2DetailingModel:
    """Resolve the official detailing metadata and patch the Stage-2 model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (
                    lora_choices(),
                    {
                        "default": OFFICIAL_DETAILING_LORA_BASENAME,
                        "tooltip": "Official LTX-2.5 x2 Pixel-Spatial detailing IC-LoRA.",
                    },
                ),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": OFFICIAL_DETAILING_STRENGTH,
                        "min": -20.0,
                        "max": 20.0,
                        "step": 0.05,
                        "tooltip": "Official Spatial DFR strength is 0.5.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "LTX_DFR_DETAILING_SPEC")
    RETURN_NAMES = ("stage_2_detailing_model", "detailing_spec")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - Detailing LoRA"
    DESCRIPTION = (
        "Resolves reference_downscale_factor from the selected official IC-LoRA metadata and applies the "
        "LoRA to a cloned MODEL in one model-only operation. It is independent of Stage-2 video conditioning."
    )

    def prepare(self, model, lora_name, strength_model=OFFICIAL_DETAILING_STRENGTH):
        return prepare_stage2_detailing_model(model, lora_name, strength_model)


class LTXDFRFinalizeStage2DFRConditioning:
    """Append the seeded DFR slots and half-res IC-LoRA reference for Stage 2."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscaled_stage_1_handoff": ("LTX_DFR_UPSCALED_STAGE1_HANDOFF",),
                "detailing_spec": ("LTX_DFR_DETAILING_SPEC",),
                "video_after_user_conditions": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Start from Stage 2B upscaled_video_latent. Re-encode the original user images at full "
                            "Stage-2 resolution, then apply the SAME official VideoConditionByLatentIndex / "
                            "VideoConditionByKeyframeIndex nodes used in Stage 1. If there are no user image "
                            "conditions, connect upscaled_video_latent directly."
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("stage_2_video_state",)
    FUNCTION = "finalize"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - Conditioning"
    DESCRIPTION = (
        "Stage 2C. Finishes the official Stage-2 conditioning list AFTER ordinary user image/keyframe "
        "conditions have been reapplied at full resolution. It first appends VideoGeneratedKeyframeSlots "
        "seeded by the Stage-1 upscaled keyframes, then appends VideoConditionByReferenceLatent using the "
        "reserved half-resolution Stage-1 video at the official x2 reference factor. No LoRA/model/noising occurs here."
    )

    def finalize(self, upscaled_stage_1_handoff, detailing_spec, video_after_user_conditions):
        dfr_layout = layout_from_handoff(upscaled_stage_1_handoff)
        stage_2_video_state = assemble_stage2_conditioning_from_upscaled_handoff(
            upscaled_stage_1_handoff=upscaled_stage_1_handoff,
            detailing_spec=detailing_spec,
            video_after_user_conditions=video_after_user_conditions,
            dfr_layout=dfr_layout,
        )
        return (stage_2_video_state,)


class LTXDFRValidateStage2DFRConditioning:
    """Validate the complete Stage-2C video token-state before any Stage-2 model call."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": ("LTX_DFR_STAGE2_HANDOFF",),
                "detailing_spec": ("LTX_DFR_DETAILING_SPEC",),
                "stage_2_video_state": ("LATENT",),
                "video_after_user_conditions": ("LATENT",),
                "upscaled_generated_keyframes": ("LATENT",),
                "reserved_half_res_video": ("LATENT",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "state_error",
        "slot_seed_error",
        "reference_error",
        "layout_error",
        "ordering_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Rebuilds Stage 2C independently and checks the complete official token state. In particular it proves "
        "the deferred initial_keyframes case: upscaled Stage-1 keyframes must occupy the generated-slot current "
        "latent exactly, with clean=0, denoise_mask=1 and keyframes_mask=1; the reference tail must contain the "
        "exact reserved half-resolution Stage-1 video with strength 1 and no keyframe marker."
    )

    def validate(
        self,
        stage_2_handoff,
        detailing_spec,
        stage_2_video_state,
        video_after_user_conditions,
        upscaled_generated_keyframes,
        reserved_half_res_video,
        dfr_layout,
    ):
        r = validate_stage2_conditioning(
            stage_2_handoff=stage_2_handoff,
            detailing_spec=detailing_spec,
            stage_2_video_state=stage_2_video_state,
            video_after_user_conditions=video_after_user_conditions,
            upscaled_generated_keyframes=upscaled_generated_keyframes,
            reserved_half_res_video=reserved_half_res_video,
            dfr_layout=dfr_layout,
        )
        report = (
            f"PASS={r['passed']}; state_error={r['state_error']:.9g}; slot_seed_error={r['slot_seed_error']:.9g}; "
            f"reference_error={r['reference_error']:.9g}; layout_error={r['layout_error']:.9g}; "
            f"ordering_error={r['ordering_error']:.9g}; slot_clean_zero_error={r['slot_clean_zero_error']:.9g}; "
            f"slot_mask_error={r['slot_mask_error']:.9g}; slot_keyframe_mask_error={r['slot_keyframe_mask_error']:.9g}; "
            f"reference_mask_error={r['reference_mask_error']:.9g}; "
            f"reference_keyframe_mask_error={r['reference_keyframe_mask_error']:.9g}; "
            f"token_latent_error={r['token_latent_error']:.9g}; token_clean_error={r['token_clean_error']:.9g}; "
            f"token_mask_error={r['token_mask_error']:.9g}; token_position_error={r['token_position_error']:.9g}; "
            f"user_conditionings={r['user_conditionings']}; positions={r['positions']}; "
            f"reference_downscale_factor={r['reference_downscale_factor']}; target_shape={r['target_shape']}; "
            f"slot_shape={r['slot_shape']}; reference_shape={r['reference_shape']}; total_tokens={r['total_tokens']}; "
            f"operation_order={r['operation_order']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                r['passed'],
                r['state_error'],
                r['slot_seed_error'],
                r['reference_error'],
                r['layout_error'],
                r['ordering_error'],
                report,
            ),
        }


class LTXDFRApplyStage2DetailingICLoRA:
    """Apply the official LTX-2.5 x2 Pixel-Spatial IC-LoRA to a cloned Comfy MODEL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "stage_2_video_state": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Validated Stage-2C video state. This is used only to verify that its final "
                            "VideoConditionByReferenceLatent downscale factor matches the IC-LoRA metadata."
                        )
                    },
                ),
                "detailing_spec": (
                    "LTX_DFR_DETAILING_SPEC",
                    {"tooltip": "Connect the spec from 'Resolve Stage 2 Detailing Metadata'."},
                ),
                "strength_model": (
                    "FLOAT",
                    {
                        "default": OFFICIAL_DETAILING_STRENGTH,
                        "min": -20.0,
                        "max": 20.0,
                        "step": 0.05,
                        "tooltip": (
                            "Detailing IC-LoRA model strength. Current upstream DFR hardcodes 0.5. "
                            "The input remains exposed so 0.5 can be compared directly with other values."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("stage_2_detailing_model", "report")
    FUNCTION = "apply"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    DESCRIPTION = (
        "Stage 2D. Uses Comfy's native LoraLoaderModelOnly to clone the Stage-1/base MODEL and add the "
        "official LTX-2.5 x2 Pixel-Spatial IC-LoRA. Strength is exposed as an input and defaults to the "
        "current upstream DFR value 0.5. It then validates the "
        "LoRA metadata reference_downscale_factor=2 against the Stage-2C reference conditioning. "
        "No noising or transformer execution occurs here."
    )

    def apply(self, model, stage_2_video_state, detailing_spec, strength_model=OFFICIAL_DETAILING_STRENGTH):
        patched_model, _factor, report = apply_stage2_detailing_lora(
            model=model,
            stage_2_video_state=stage_2_video_state,
            detailing_spec=detailing_spec,
            strength_model=strength_model,
        )
        return (patched_model, report)


class LTXDFRValidateStage2DetailingICLoRA:
    """Validate the Stage-2D model patch and IC-LoRA metadata before sampling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("MODEL",),
                "stage_2_detailing_model": ("MODEL",),
                "stage_2_video_state": ("LATENT",),
                "detailing_spec": ("LTX_DFR_DETAILING_SPEC",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "metadata_error",
        "reference_factor_error",
        "uuid_error",
        "patch_count_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Checks that Stage 2D used the official x2 Pixel-Spatial IC-LoRA at current upstream strength 0.5, "
        "that the native "
        "LoRA metadata resolves to reference_downscale_factor=2, that Stage-2C uses the same factor, and "
        "that Comfy produced a cloned ModelPatcher with an expanded/different LoRA patch set."
    )

    def validate(self, base_model, stage_2_detailing_model, stage_2_video_state, detailing_spec):
        r = validate_stage2_detailing_lora(
            base_model=base_model,
            detailing_model=stage_2_detailing_model,
            stage_2_video_state=stage_2_video_state,
            detailing_spec=detailing_spec,
        )
        report = (
            f"PASS={r['passed']}; metadata_error={r['metadata_error']:.9g}; "
            f"reference_factor_error={r['reference_factor_error']:.9g}; uuid_error={r['uuid_error']:.9g}; "
            f"patch_count_error={r['patch_count_error']:.9g}; contract_error={r['contract_error']:.9g}; "
            f"lora_name_error={r['lora_name_error']:.9g}; contract_lora_error={r['contract_lora_error']:.9g}; "
            f"strength_error={r['strength_error']:.9g}; same_object_error={r['same_object_error']:.9g}; "
            f"underlying_model_error={r['underlying_model_error']:.9g}; "
            f"lora={r['lora_basename']}; strength_model={r['strength_model']}; "
            f"reference_downscale_factor={r['reference_downscale_factor']}; "
            f"stage2_reference_factor={r['stage2_reference_factor']}; "
            f"base_patch_count={r['base_patch_count']}; detailing_patch_count={r['detailing_patch_count']}; "
            f"native_node={r['native_node']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                r['passed'],
                r['metadata_error'],
                r['reference_factor_error'],
                r['uuid_error'],
                r['patch_count_error'],
                report,
            ),
        }


class LTXDFRStage2AVGaussianNoiser:
    """Continue the shared Stage-1 RNG stream and noise Stage-2 video then audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": ("LTX_DFR_STAGE2_HANDOFF",),
                "stage_2_video_state": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Validated Stage-2C video conditioning state. The authoritative token state already "
                            "contains the upscaled base video, full-res user conditions, seeded DFR slots and "
                            "half-res reference latent."
                        )
                    },
                ),
                "stage_1_audio_for_stage2": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Stage-1 audio latent carried through Stage 2B unchanged. Upstream uses this as the "
                            "Stage-2 audio initial latent before re-noising."
                        )
                    },
                ),
                "stage_2_sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Exact official Stage-2 DFR sigma schedule. Stage 2E uses sigma[0]=0.909375 as the "
                            "noise_scale for BOTH video and audio."
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("noised_stage_2_video_state", "noised_stage_2_audio_state", "report")
    FUNCTION = "noise"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - AV Noiser"
    DESCRIPTION = (
        "Stage 2E. Restores the exact torch.Generator continuation captured after Stage-1 video+audio noising, "
        "then mirrors upstream GaussianNoiser ordering: Stage-2 VIDEO draw first, Stage-2 AUDIO draw second. "
        "Both modalities use noise_scale=stage_2_sigmas[0] (0.909375). It never reseeds."
    )

    def noise(self, stage_2_handoff, stage_2_video_state, stage_1_audio_for_stage2, stage_2_sigmas):
        return materialize_stage2_av_gaussian_noised_states(
            stage_2_handoff=stage_2_handoff,
            stage_2_video_state=stage_2_video_state,
            stage_1_audio_for_stage2=stage_1_audio_for_stage2,
            stage_2_sigmas_tensor=stage_2_sigmas,
        )


class LTXDFRValidateStage2AVGaussianNoiser:
    """Independently replay Stage-1 + Stage-2 RNG consumption and verify Stage-2 noising."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": ("LTX_DFR_STAGE2_HANDOFF",),
                "pre_noise_stage_2_video_state": ("LATENT",),
                "stage_1_audio_for_stage2": ("LATENT",),
                "stage_2_sigmas": ("SIGMAS",),
                "noised_stage_2_video_state": ("LATENT",),
                "noised_stage_2_audio_state": ("LATENT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "video_error",
        "audio_error",
        "rng_start_error",
        "rng_end_error",
        "metadata_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Replays the RNG independently from the original seed: consumes the saved Stage-1 video/audio noise "
        "shapes, then Stage-2 video/audio draws, and checks exact tensors plus the GaussianNoiser formula, "
        "masks, clean latents and continuation metadata."
    )

    def validate(
        self,
        stage_2_handoff,
        pre_noise_stage_2_video_state,
        stage_1_audio_for_stage2,
        stage_2_sigmas,
        noised_stage_2_video_state,
        noised_stage_2_audio_state,
    ):
        r = validate_stage2_av_gaussian_noiser(
            stage_2_handoff=stage_2_handoff,
            pre_noise_stage_2_video_state=pre_noise_stage_2_video_state,
            stage_1_audio_for_stage2=stage_1_audio_for_stage2,
            stage_2_sigmas_tensor=stage_2_sigmas,
            noised_stage_2_video_state=noised_stage_2_video_state,
            noised_stage_2_audio_state=noised_stage_2_audio_state,
        )
        report = (
            f"PASS={r['passed']}; video_error={r['video_error']:.9g}; audio_error={r['audio_error']:.9g}; "
            f"video_clean_error={r['video_clean_error']:.9g}; audio_clean_error={r['audio_clean_error']:.9g}; "
            f"video_mask_error={r['video_mask_error']:.9g}; audio_mask_error={r['audio_mask_error']:.9g}; "
            f"rng_start_error={r['rng_start_error']:.9g}; rng_end_error={r['rng_end_error']:.9g}; "
            f"metadata_error={r['metadata_error']:.9g}; "
            f"stage1_audio_initial_error={r['stage1_audio_initial_error']:.9g}; "
            f"seed={r['seed']}; seed_mode=continued_not_reseeded; noise_scale={r['noise_scale']:.9g}; "
            f"video_token_shape={r['video_token_shape']}; audio_token_shape={r['audio_token_shape']}; "
            f"video_clean_tokens={r['video_clean_tokens']}; video_noised_tokens={r['video_noised_tokens']}; "
            f"device={r['device']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                r['passed'],
                r['video_error'],
                r['audio_error'],
                r['rng_start_error'],
                r['rng_end_error'],
                r['metadata_error'],
                report,
            ),
        }



class LTXDFRExecuteStage2AVDenoise:
    """Run one real Stage-2 AV transformer/x0 pass using the detailing IC-LoRA MODEL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Connect the MODEL from 'LTX DFR: Apply Stage 2 Detailing IC-LoRA'. "
                            "Strict Stage 2F parity intentionally requires the detail-enabled model."
                        )
                    },
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "noised_stage_2_video_state": ("LATENT",),
                "noised_stage_2_audio_state": ("LATENT",),
                "stage_2_sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Exact official Stage-2 DFR sigma schedule. Stage 2F executes one raw x0 pass "
                            "at stage_2_sigmas[step_index]."
                        )
                    },
                ),
                "step_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("raw_denoised_stage_2_video_state", "raw_denoised_stage_2_audio_state", "report")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - AV Execution"
    DESCRIPTION = (
        "Stage 2F. Materializes the noised Stage-2 VIDEO/AUDIO official states into native Comfy/LTXAV input "
        "geometry, runs one real transformer/x0 evaluation through the Stage-2 detailing MODEL, then repacks "
        "the output back into the strict official token-state format. This is the Stage-2 counterpart of "
        "'Execute Stage 1 AV Denoise'."
    )

    def run(self, model, positive, negative, noised_stage_2_video_state, noised_stage_2_audio_state, stage_2_sigmas, step_index=0, cfg_scale=1.0):
        denoised_video_state, denoised_audio_state, _av_model_input, report = execute_stage2_av_denoise(
            model=model,
            positive=positive,
            negative=negative,
            noised_stage_2_video_state=noised_stage_2_video_state,
            noised_stage_2_audio_state=noised_stage_2_audio_state,
            sigmas=stage_2_sigmas,
            step_index=step_index,
            cfg_scale=cfg_scale,
            seed=None,
        )
        return (denoised_video_state, denoised_audio_state, report)


class LTXDFRValidateStage2AVDenoise:
    """Validate the structural correctness of one raw Stage-2 AV denoise output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_stage_2_video_state": ("LATENT",),
                "noised_stage_2_audio_state": ("LATENT",),
                "raw_denoised_stage_2_video_state": ("LATENT",),
                "raw_denoised_stage_2_audio_state": ("LATENT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "video_shape_error",
        "audio_shape_error",
        "video_preview_error",
        "audio_preview_error",
        "metadata_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Checks that the raw Stage-2 AV denoise output preserves VIDEO/AUDIO token geometry, can be rebuilt "
        "exactly into preview latents, contains only finite values, and that the connected inputs really are "
        "Stage-2 shared-AV noiser states."
    )

    def validate(self, noised_stage_2_video_state, noised_stage_2_audio_state, raw_denoised_stage_2_video_state, raw_denoised_stage_2_audio_state):
        r = validate_stage2_av_denoise(
            noised_stage_2_video_state=noised_stage_2_video_state,
            noised_stage_2_audio_state=noised_stage_2_audio_state,
            raw_denoised_stage_2_video_state=raw_denoised_stage_2_video_state,
            raw_denoised_stage_2_audio_state=raw_denoised_stage_2_audio_state,
        )
        report = (
            f"PASS={r['passed']}; video_shape_error={r['video_shape_error']:.9g}; "
            f"audio_shape_error={r['audio_shape_error']:.9g}; video_preview_error={r['video_preview_error']:.9g}; "
            f"audio_preview_error={r['audio_preview_error']:.9g}; video_finite_error={r['video_finite_error']:.9g}; "
            f"audio_finite_error={r['audio_finite_error']:.9g}; video_delta={r['video_delta']:.9g}; "
            f"audio_delta={r['audio_delta']:.9g}; metadata_error={r['metadata_error']:.9g}; seed={r['seed']}; "
            f"video_token_shape={r['video_token_shape']}; audio_token_shape={r['audio_token_shape']}; "
            f"video_device={r['video_device']}; audio_device={r['audio_device']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                r['passed'],
                r['video_shape_error'],
                r['audio_shape_error'],
                r['video_preview_error'],
                r['audio_preview_error'],
                r['metadata_error'],
                report,
            ),
        }


class LTXDFRStage2AVEulerStepFromDenoised:
    """Apply exact official post-process + deterministic Euler to both Stage-2 modalities."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_stage_2_video_state": ("LATENT",),
                "noised_stage_2_audio_state": ("LATENT",),
                "raw_denoised_stage_2_video_state": ("LATENT",),
                "raw_denoised_stage_2_audio_state": ("LATENT",),
                "stage_2_sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Exact official Stage-2 DFR sigma schedule. Stage 2G applies post-process and one "
                            "Euler transition from stage_2_sigmas[step_index] to stage_2_sigmas[step_index+1]."
                        )
                    },
                ),
                "step_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT", "STRING")
    RETURN_NAMES = (
        "next_stage_2_video_state",
        "next_stage_2_audio_state",
        "post_processed_stage_2_video_state",
        "post_processed_stage_2_audio_state",
        "report",
    )
    FUNCTION = "step"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - AV Execution"
    DESCRIPTION = (
        "Stage 2G. Mirrors the official LTX joint euler_denoising_loop after one Stage-2 AV transformer "
        "evaluation: post_process_latent is applied independently to VIDEO and AUDIO using each modality's "
        "clean_latent and denoise_mask, then EulerDiffusionStep advances both from "
        "stage_2_sigmas[step_index] to stage_2_sigmas[step_index+1]."
    )

    def step(
        self,
        noised_stage_2_video_state,
        noised_stage_2_audio_state,
        raw_denoised_stage_2_video_state,
        raw_denoised_stage_2_audio_state,
        stage_2_sigmas,
        step_index=0,
    ):
        return stage2_av_euler_step_from_denoised(
            noised_stage_2_video_state,
            noised_stage_2_audio_state,
            raw_denoised_stage_2_video_state,
            raw_denoised_stage_2_audio_state,
            stage_2_sigmas,
            step_index,
        )


class LTXDFRValidateStage2AVEulerStep:
    """Exact-value validator for the Stage-2 joint AV post-process + Euler transition."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_stage_2_video_state": ("LATENT",),
                "noised_stage_2_audio_state": ("LATENT",),
                "raw_denoised_stage_2_video_state": ("LATENT",),
                "raw_denoised_stage_2_audio_state": ("LATENT",),
                "next_stage_2_video_state": ("LATENT",),
                "next_stage_2_audio_state": ("LATENT",),
                "post_processed_stage_2_video_state": ("LATENT",),
                "post_processed_stage_2_audio_state": ("LATENT",),
                "stage_2_sigmas": ("SIGMAS",),
                "step_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "video_next_error",
        "audio_next_error",
        "video_post_error",
        "audio_post_error",
        "metadata_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Independently recomputes the exact Stage-2 VIDEO and AUDIO post-process/Euler transition, checks all "
        "latent values, and verifies preservation of clean_latent, denoise_mask and positions."
    )

    def validate(
        self,
        noised_stage_2_video_state,
        noised_stage_2_audio_state,
        raw_denoised_stage_2_video_state,
        raw_denoised_stage_2_audio_state,
        next_stage_2_video_state,
        next_stage_2_audio_state,
        post_processed_stage_2_video_state,
        post_processed_stage_2_audio_state,
        stage_2_sigmas,
        step_index=0,
    ):
        r = validate_stage2_av_euler_step(
            noised_stage_2_video_state,
            noised_stage_2_audio_state,
            raw_denoised_stage_2_video_state,
            raw_denoised_stage_2_audio_state,
            next_stage_2_video_state,
            next_stage_2_audio_state,
            post_processed_stage_2_video_state,
            post_processed_stage_2_audio_state,
            stage_2_sigmas,
            step_index,
        )
        report = (
            f"PASS={r['passed']}; sigma={r['sigma']:.9g}; sigma_next={r['sigma_next']:.9g}; "
            f"video_next_error={r['video_next_error']:.9g}; audio_next_error={r['audio_next_error']:.9g}; "
            f"video_post_error={r['video_post_error']:.9g}; audio_post_error={r['audio_post_error']:.9g}; "
            f"video_mask_error={r['video_mask_error']:.9g}; audio_mask_error={r['audio_mask_error']:.9g}; "
            f"video_position_error={r['video_position_error']:.9g}; audio_position_error={r['audio_position_error']:.9g}; "
            f"video_clean_error={r['video_clean_error']:.9g}; audio_clean_error={r['audio_clean_error']:.9g}; "
            f"metadata_error={r['metadata_error']:.9g}; seed={r['seed']}; "
            f"video_token_shape={r['video_token_shape']}; audio_token_shape={r['audio_token_shape']}."
        )
        return {"ui": {"text": [report]}, "result": (
            r['passed'],
            r['video_next_error'],
            r['audio_next_error'],
            r['video_post_error'],
            r['audio_post_error'],
            r['metadata_error'],
            report,
        )}


class LTXDFRRunStage2AVLoop:
    """Run the complete official deterministic Stage-2 AV trajectory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Connect the MODEL from 'LTX DFR: Prepare Stage 2 Detailing Model'."
                        )
                    },
                ),
                "positive": ("CONDITIONING",),
                "upscaled_stage_1_handoff": ("LTX_DFR_UPSCALED_STAGE1_HANDOFF",),
                "stage_2_video_state": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Validated Stage-2C video state after full-resolution user conditions, generated-slot seeding, "
                            "and half-resolution reference conditioning."
                        )
                    },
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Keep 1.0 for the distilled official-parity Stage-2 AV test.",
                    },
                ),
            },
            "optional": {
                "compact_timesteps": ("BOOLEAN", {"default": True, "tooltip": "Store repeated video timestep conditioning once with exact token indices. Disable to use native preparation. Batch >1 and highly varying timesteps use the native fallback."}),
                "output_chunk_tokens": ("INT", {"default": 4096, "min": 0, "max": 65536, "step": 1024, "tooltip": "Chunk video output modulation/projection to reduce peak VRAM. 0 disables. Timestep conditioning is unchanged."}),
                "diagnose_memory": ("BOOLEAN", {"default": False, "tooltip": "Diagnostic run: verify executed FF chunks and measure attention/FF memory peaks. Adds eager boundaries and synchronization; disable for speed comparisons."}),
                "ff_chunk_tokens": ("INT", {"default": 4096, "min": 0, "max": 65536, "step": 1024, "tooltip": "Video feed-forward chunk size. 4096 reduces peak VRAM; 0 disables. Smaller chunks may increase weight-transfer overhead."}),
                "negative": (
                    "CONDITIONING",
                    {"tooltip": "Optional Comfy CFG extension. Strict official parity does not require it."},
                ),
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Optional advanced schedule override. When disconnected, the exact official "
                            "four-point / three-transition Stage-2 schedule is used."
                        )
                    },
                ),
                "temporal_tiles": ("INT", {"default": 1, "min": 0, "max": 64, "step": 1, "tooltip": "1: original untiled Stage 2. 0: inherit temporal window count from modular handoff. 2 or more: split model predictions in time only, combining each step before the shared AV update. Full spatial resolution is retained."}),
            },
        }

    RETURN_TYPES = ("LTX_DFR_STAGE2_RESULT_HANDOFF",)
    RETURN_NAMES = ("stage_2_handoff",)
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - AV Execution"
    DESCRIPTION = (
        "Stage 2H. Runs the complete detail-enabled official Stage-2 AV trajectory: continue the Stage-1 RNG stream "
        "for the initial Stage-2 Gaussian noiser, then run the validated real AV denoise bridge and joint post-process "
        "+ Euler step for the complete schedule. When sigmas is disconnected it uses the official three-transition "
        "schedule; a connected valid custom schedule runs one transition per adjacent sigma pair. The single output "
        "contains the final video, preserved Stage-1 "
        "audio, generated decoder keyframes and continued decoder RNG state without retaining the large token states."
    )

    def run(
        self,
        model,
        positive,
        upscaled_stage_1_handoff,
        stage_2_video_state,
        cfg_scale=1.0,
        negative=None,
        sigmas=None,
        ff_chunk_tokens=4096,
        diagnose_memory=False,
        output_chunk_tokens=4096,
        compact_timesteps=True,
        temporal_tiles=1,
    ):
        return (
            run_stage2_spatial_dfr(
                model=model,
                positive=positive,
                negative=negative,
                upscaled_stage_1_handoff=upscaled_stage_1_handoff,
                stage_2_video_state=stage_2_video_state,
                sigmas=sigmas,
                cfg_scale=float(cfg_scale),
                ff_chunk_tokens=int(ff_chunk_tokens),
                diagnose_memory=bool(diagnose_memory),
                output_chunk_tokens=int(output_chunk_tokens),
                compact_timesteps=bool(compact_timesteps),
                temporal_tiles=temporal_tiles,
            ),
        )


class LTXExperimentalRunStage2AVLoopExtraStep:
    """Run the controlled four-transition Stage-2 comparison."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Connect the MODEL from 'LTX DFR: Prepare Stage 2 Detailing Model'."
                        )
                    },
                ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "upscaled_stage_1_handoff": ("LTX_DFR_UPSCALED_STAGE1_HANDOFF",),
                "stage_2_video_state": (
                    "LATENT",
                    {
                        "tooltip": (
                            "The same validated Stage-2C video state used by the official Stage-2 loop."
                        )
                    },
                ),
                "stage_2_sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Connect the official 4-point Stage-2 schedule. This experimental node verifies it, "
                            "then inserts 0.2109375 between 0.421875 and 0 internally."
                        )
                    },
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Keep 1.0 so only the added Stage-2 transition changes.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("final_stage_2_video_state", "final_stage_2_audio_state", "stage_2_base_latent", "report")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/Experimental"
    DESCRIPTION = (
        "Non-parity Stage 2H comparison. Preserves the official start sigma and first two transitions, then splits "
        "the largest final interval: 0.421875 -> 0.2109375 -> 0. This performs four Stage-2 model evaluations "
        "instead of three; every other Stage-2 input and the C17c decode path remain unchanged."
    )

    def run(self, model, positive, negative, upscaled_stage_1_handoff, stage_2_video_state, stage_2_sigmas, cfg_scale=1.0):
        return execute_experimental_stage2_av_denoising_loop_extra_step_from_upscaled_handoff(
            model=model,
            positive=positive,
            negative=negative,
            upscaled_stage_1_handoff=upscaled_stage_1_handoff,
            stage_2_video_state=stage_2_video_state,
            official_stage_2_sigmas=stage_2_sigmas,
            cfg_scale=cfg_scale,
        )


class LTXDFRValidateStage2AVLoop:
    """Validate the exported outputs of the full Stage-2 AV loop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_stage_2_video_state": ("LATENT",),
                "final_stage_2_audio_state": ("LATENT",),
                "stage_2_base_latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "base_max_error",
        "audio_preview_error",
        "metadata_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Checks that the full Stage-2 AV loop exported the final base video latent as an exact slice of the final "
        "strict official video state, that the final audio state still round-trips exactly from its token view, and "
        "that the final states still carry valid Stage-2 shared-AV metadata."
    )

    def validate(self, final_stage_2_video_state, final_stage_2_audio_state, stage_2_base_latent):
        r = validate_stage2_av_loop_outputs(
            final_stage_2_video_state,
            final_stage_2_audio_state,
            stage_2_base_latent,
        )
        report = (
            f"PASS={r['passed']}; base_max_error={r['base_max_abs_error']:.9g}; "
            f"audio_preview_error={r['audio_preview_error']:.9g}; metadata_error={r['metadata_error']:.9g}; "
            f"shared_seed={r['shared_seed']}; video_token_shape={r['video_token_shape']}; "
            f"base_latent_shape={r['base_latent_shape']}; audio_shape={r['audio_shape']}; "
            f"audio_token_shape={r['audio_token_shape']}; duration_seconds={r['duration_seconds']:.9g}."
        )
        return {"ui": {"text": [report]}, "result": (
            r['passed'],
            r['base_max_abs_error'],
            r['audio_preview_error'],
            r['metadata_error'],
            report,
        )}


class LTXSpatialDFRStage1PreviewDecode:
    """Keyframe-aware DiffVAE preview at the frozen Stage-1 boundary."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths

            vae_choices = folder_paths.get_filename_list("vae")
        except Exception:
            vae_choices = []
        return {
            "required": {
                "stage_1_handoff": (
                    "LTX_DFR_STAGE2_HANDOFF",
                    {"tooltip": "Single output from 'Run Stage 1 Spatial DFR'."},
                ),
                "final_video_latent": (
                    "LATENT",
                    {"tooltip": "Video output from 'Split Stage 1 Audio / Video'."},
                ),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
                "vae": ("VAE",),
                "vae_name": (
                    vae_choices,
                    {
                        "tooltip": (
                            "Select the same video VAE checkpoint loaded by the connected VAE. "
                            "The decoder type_emb is recovered from that checkpoint."
                        )
                    },
                ),
                "use_auto_tiling": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use the official memory-aware tile recommender. Manual tile sizes are ignored.",
                    },
                ),
                "tile_frames": (
                    "INT",
                    {
                        "default": 104,
                        "min": 80,
                        "max": 10000,
                        "step": 8,
                        "tooltip": "Manual mode only. Temporal overlap is derived internally (currently 40).",
                    },
                ),
                "tile_height": (
                    "INT",
                    {
                        "default": 416,
                        "min": 320,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "Manual mode only. Spatial overlap is derived internally (currently 160).",
                    },
                ),
                "tile_width": (
                    "INT",
                    {
                        "default": 544,
                        "min": 320,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "Manual mode only. Spatial overlap is derived internally (currently 160).",
                    },
                ),
                "attention_chunks": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Allowed: 1, 2, or 4. One is fastest in measured runs; 2/4 reduce attention workspace. Independent of auto/manual tiling."}),
                "auto_tile_multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05,
                    "tooltip": "Auto tiling only. Multiplies both computed tile caps. 1.0 preserves current behavior; below 1 uses smaller limits; above 1 allows larger tiles. Existing memory checks still apply."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "LTX/DFR Spatial/Stage 1"
    DESCRIPTION = (
        "Creates an improved keyframe-aware DiffVAE preview directly from the frozen Stage-1 result. It restores "
        "the post-Stage-1 AV RNG state and decodes the Stage-1 generated slots through the same validated "
        "deterministic, Stage-5 and auto/manual tiled path used by the final Stage-2 decoder. This is a preview "
        "extension; official Spatial DFR still continues through Stage 2 before its final output."
    )

    def decode(
        self,
        stage_1_handoff,
        final_video_latent,
        dfr_layout,
        vae,
        vae_name,
        use_auto_tiling=True,
        tile_frames=104,
        tile_height=416,
        tile_width=544,
        attention_chunks=1,
        auto_tile_multiplier=1.0,
    ):
        images = decode_stage1_spatial_dfr_video(
            stage_1_handoff,
            final_video_latent,
            dfr_layout,
            vae,
            str(vae_name),
            use_auto_tiling=bool(use_auto_tiling),
            tile_frames=int(tile_frames),
            tile_height=int(tile_height),
            tile_width=int(tile_width),
            attention_chunks=attention_chunks,
            auto_tile_multiplier=auto_tile_multiplier,
        )
        return (images,)


class LTXDFRSpatialDFRVideoDecode:
    """Single production node for the complete keyframe-aware Spatial-DFR decode."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths

            vae_choices = folder_paths.get_filename_list("vae")
        except Exception:
            vae_choices = []
        return {
            "required": {
                "stage_2_handoff": (
                    "LTX_DFR_STAGE2_RESULT_HANDOFF",
                    {"tooltip": "Single output from 'Run Stage 2 AV Loop'."},
                ),
                "final_video_latent": (
                    "LATENT",
                    {"tooltip": "Video output from 'Finalize Stage 2 Output / Trim'."},
                ),
                "vae": ("VAE",),
                "vae_name": (
                    vae_choices,
                    {
                        "tooltip": (
                            "Select the same video VAE checkpoint loaded by the connected VAE. "
                            "The official decoder type_emb must be recovered from that checkpoint."
                        )
                    },
                ),
                "use_auto_tiling": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use the official memory-aware tile recommender. Manual tile sizes are ignored.",
                    },
                ),
                "tile_frames": (
                    "INT",
                    {
                        "default": 104,
                        "min": 80,
                        "max": 10000,
                        "step": 8,
                        "tooltip": "Manual mode only. Temporal overlap is derived internally (currently 40).",
                    },
                ),
                "tile_height": (
                    "INT",
                    {
                        "default": 416,
                        "min": 320,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "Manual mode only. Spatial overlap is derived internally (currently 160).",
                    },
                ),
                "tile_width": (
                    "INT",
                    {
                        "default": 544,
                        "min": 320,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "Manual mode only. Spatial overlap is derived internally (currently 160).",
                    },
                ),
                "attention_chunks": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Allowed: 1, 2, or 4. One is fastest in measured runs; 2/4 reduce attention workspace. Independent of auto/manual tiling."}),
                "auto_tile_multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05,
                    "tooltip": "Auto tiling only. Multiplies both computed tile caps. 1.0 preserves current behavior; below 1 uses smaller limits; above 1 allows larger tiles. Existing memory checks still apply."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - Final Output"
    DESCRIPTION = (
        "Runs the complete official keyframe-aware Spatial-DFR video decode internally: final keyframes and continued RNG, "
        "trained decoder type_emb, Triton joint attention, deterministic stages, Stage 5, tiling and pixel assembly. "
        "Auto tiling is the default; manual mode uses the three tile sizes with checkpoint-derived overlaps."
    )

    def decode(
        self,
        stage_2_handoff,
        final_video_latent,
        vae,
        vae_name,
        use_auto_tiling=True,
        tile_frames=104,
        tile_height=416,
        tile_width=544,
        attention_chunks=1,
        auto_tile_multiplier=1.0,
    ):
        dfr_layout = layout_from_handoff(stage_2_handoff)
        images = decode_spatial_dfr_video(
            stage_2_handoff,
            final_video_latent,
            dfr_layout,
            vae,
            str(vae_name),
            use_auto_tiling=bool(use_auto_tiling),
            tile_frames=int(tile_frames),
            tile_height=int(tile_height),
            tile_width=int(tile_width),
            attention_chunks=attention_chunks,
            auto_tile_multiplier=auto_tile_multiplier,
        )
        return (images,)


class LTXDFRExperimentalBlockStreamingDecode:
    @classmethod
    def INPUT_TYPES(cls):
        required=LTXDFRSpatialDFRVideoDecode.INPUT_TYPES()["required"]
        required={k:v for k,v in required.items() if k in ("stage_2_handoff","final_video_latent","vae","vae_name")}
        required.update({
            "tile_frames": ("INT", {"default":104,"min":80,"max":10000,"step":8,"tooltip":"Temporal window size in frames. Uses the original checkpoint-derived overlap (currently 40 frames)."}),
            "core_height": ("INT", {"default":320,"min":32,"max":2048,"step":8,"tooltip":"Owned output core in pixels; halos added internally. Smaller cores reduce GPU workspace."}),
            "core_width": ("INT", {"default":512,"min":32,"max":2048,"step":8,"tooltip":"Owned output core in pixels. Start at 320x512 on a 12GB GPU."}),
        })
        temporal=required.pop("tile_frames")
        required["tile_frames"]=temporal
        required["use_auto_tiling"]=("BOOLEAN", {"default":False,"tooltip":"Automatically select temporal window and spatial cores using the classic autotiler, adjusted for streaming halos. Ignores manual core and frame settings."})
        required["auto_tile_multiplier"]=("FLOAT", {"default":1.0,"min":0.1,"max":2.0,"step":0.05,"tooltip":"Auto mode only. Scales the classic autotiler token and spatial-area caps; lower permits smaller tiles, higher permits larger tiles."})
        return {"required":required}
    RETURN_TYPES=("IMAGE","STRING")
    RETURN_NAMES=("images","report")
    FUNCTION="decode"
    CATEGORY="LTX/DFR Spatial/Experimental"
    DESCRIPTION="Experimental CPU-backed eight-block DiffVAE decode. Connect the same Stage-2 handoff, final latent and VAE as Spatial DFR Video Decode. Layout is read from the handoff. Uses original temporal windows and global noise within each window; outputs can differ. Requires substantial system RAM (roughly 20GB+ for a 97-frame 2MP case). Run separately from the baseline."
    def decode(self,stage_2_handoff,final_video_latent,vae,vae_name,core_height=320,core_width=512,tile_frames=104,use_auto_tiling=False,auto_tile_multiplier=1.0):
        dfr_layout = layout_from_handoff(stage_2_handoff)
        from .decoder_block_streaming_node import decode_experimental
        return decode_experimental(stage_2_handoff,final_video_latent,dfr_layout,vae,vae_name,core_height=core_height,core_width=core_width,tile_frames=tile_frames,use_auto_tiling=use_auto_tiling,auto_tile_multiplier=auto_tile_multiplier)


class LTXDFRPrepareOfficialFinalDecode:
    """Expose decoder continuation data from the compact Stage-2 result."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": (
                    "LTX_DFR_STAGE2_RESULT_HANDOFF",
                    {"tooltip": "Single output from 'Run Stage 2 AV Loop'."},
                ),
            }
        }

    RETURN_TYPES = ("LTX_DFR_DECODER_HANDOFF", "LTX_DFR_DECODE_KEYFRAMES")
    RETURN_NAMES = ("decoder_handoff", "final_decode_keyframes")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - Final Output"
    DESCRIPTION = (
        "Prepares the two technical inputs used by the continued-RNG and full keyframe-aware official "
        "decoder paths. The ordinary VAE Decode path does not need this node."
    )

    def prepare(self, stage_2_handoff):
        dfr_layout = layout_from_handoff(stage_2_handoff)
        return prepare_official_final_decode(stage_2_handoff, dfr_layout)


class LTXDFRPrepareStage2DecoderHandoff:
    """U3.1: preserve final Stage-2 generated slots and continued decoder RNG state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_stage_2_video_state": (
                    "LATENT",
                    {
                        "tooltip": "Final strict Stage-2 VIDEO state from 'LTX DFR: Run Stage 2 AV Loop'."
                    },
                ),
                "final_stage_2_audio_state": (
                    "LATENT",
                    {
                        "tooltip": "Final strict Stage-2 AUDIO state from 'LTX DFR: Run Stage 2 AV Loop'."
                    },
                ),
            }
        }

    RETURN_TYPES = ("LTX_DFR_DECODER_HANDOFF", "LATENT", "STRING")
    RETURN_NAMES = ("decoder_handoff", "stage_2_generated_keyframes", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    DESCRIPTION = (
        "U3.1 structural checkpoint. Extracts the final Stage-2 generated-keyframe slots and preserves the exact "
        "shared RNG continuation state captured after Stage-2 VIDEO then AUDIO noising. It does not decode, trim "
        "keyframes, reconstruct a torch.Generator, or change the final video/audio outputs."
    )

    def prepare(self, final_stage_2_video_state, final_stage_2_audio_state):
        return prepare_stage2_decoder_handoff(
            final_stage_2_video_state=final_stage_2_video_state,
            final_stage_2_audio_state=final_stage_2_audio_state,
        )


class LTXDFRValidateStage2DecoderHandoff:
    """Validate that U3.1 captured the exact final Stage-2 slots and RNG state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_stage_2_video_state": ("LATENT",),
                "final_stage_2_audio_state": ("LATENT",),
                "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
                "stage_2_generated_keyframes": ("LATENT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "generated_keyframes_error", "rng_state_error", "metadata_error", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "U3.1 regression validator. Re-extracts the final Stage-2 generated slots and RNG continuation state "
        "independently from the final VIDEO/AUDIO states and checks the decoder handoff exactly."
    )

    def validate(self, final_stage_2_video_state, final_stage_2_audio_state, decoder_handoff, stage_2_generated_keyframes):
        r = validate_stage2_decoder_handoff(
            final_stage_2_video_state=final_stage_2_video_state,
            final_stage_2_audio_state=final_stage_2_audio_state,
            decoder_handoff=decoder_handoff,
            stage_2_generated_keyframes=stage_2_generated_keyframes,
        )
        metadata_error = max(r["seed_error"], r["device_error"], r["count_error"])
        report = (
            f"PASS={r['passed']}; generated_keyframes_error={r['generated_keyframes_error']:.9g}; "
            f"rng_state_error={r['rng_state_error']:.9g}; metadata_error={metadata_error:.9g}; "
            f"seed={r['seed']}; rng_device_type={r['rng_device_type']}; rng_state_bytes={r['rng_state_bytes']}; "
            f"generated_keyframes={r['generated_keyframe_count']}; shape={r['generated_keyframe_shape']}."
        )
        return {"ui": {"text": [report]}, "result": (
            r["passed"],
            r["generated_keyframes_error"],
            r["rng_state_error"],
            float(metadata_error),
            report,
        )}


class LTXDFRBuildFinalDecodeKeyframes:
    """U3.2: trim Stage-2 generated slots to the final decoder canvas."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder_handoff": (
                    "LTX_DFR_DECODER_HANDOFF",
                    {"tooltip": "U3.1 decoder handoff containing final Stage-2 generated slots and decoder RNG state."},
                ),
                "dfr_layout": (
                    "LTX_DFR_LAYOUT",
                    {"tooltip": "Exact DFR layout. requested_frames is the post-trim decoder canvas for temporal=0."},
                ),
            }
        }

    RETURN_TYPES = ("LTX_DFR_DECODE_KEYFRAMES", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("final_decode_keyframes", "kept_pixel_frame_indices", "dropped_pixel_frame_indices", "report")
    FUNCTION = "build"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    DESCRIPTION = (
        "U3.2 structural checkpoint. Mirrors upstream decode_keyframes_from_slots() after final canvas trim: "
        "keeps only Stage-2 generated slots whose pixel-frame positions are inside requested_frames, preserves "
        "their order, and builds the final decoder-keyframe package. It does not decode or modify final outputs."
    )

    def build(self, decoder_handoff, dfr_layout):
        return build_final_decode_keyframes(decoder_handoff, dfr_layout)


class LTXDFRValidateFinalDecodeKeyframes:
    """Validate exact U3.2 slot filtering and final decoder-keyframe construction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
                "final_decode_keyframes": ("LTX_DFR_DECODE_KEYFRAMES",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "latents_error", "indices_error", "metadata_error", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "U3.2 regression validator. Rebuilds the final decoder-keyframe package independently and checks "
        "surviving latent planes, torch.long pixel-frame indices, dropped tail positions, and metadata exactly."
    )

    def validate(self, decoder_handoff, dfr_layout, final_decode_keyframes):
        r = validate_final_decode_keyframes(decoder_handoff, dfr_layout, final_decode_keyframes)
        report = (
            f"PASS={r['passed']}; latents_error={r['latents_error']:.9g}; indices_error={r['indices_error']:.9g}; "
            f"metadata_error={r['metadata_error']:.9g}; present={r['present']}; num_frames={r['num_frames']}; "
            f"source_positions={r['source_positions']}; kept_positions={r['kept_positions']}; "
            f"dropped_positions={r['dropped_positions']}; source_keyframes={r['source_keyframe_count']}; "
            f"kept_keyframes={r['kept_keyframe_count']}; latents_shape={r['latents_shape']}."
        )
        return {"ui": {"text": [report]}, "result": (
            r["passed"], r["latents_error"], r["indices_error"], r["metadata_error"], report
        )}


class LTXDFRFinalizeStage2Output:
    """Trim the padded Stage-2 base latent and package the final decode-ready outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": (
                    "LTX_DFR_STAGE2_RESULT_HANDOFF",
                    {
                        "tooltip": (
                            "Single output from 'Run Stage 2 AV Loop'. It contains the padded final video and "
                            "the preserved official Stage-1 audio latent."
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "FLOAT")
    RETURN_NAMES = ("final_video_latent", "final_audio_latent", "fps")
    FUNCTION = "finalize"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - Final Output"
    DESCRIPTION = (
        "Stage 2I latent finalizer. Trims the Stage-2 detailed video latent back from the padded working canvas and "
        "returns the preserved Stage-1 audio latent for decoding. IMPORTANT: the audio latent itself stays untrimmed; "
        "after LTXV Audio VAE Decode, run 'LTX DFR: Trim Stage 2 Decoded Audio' to match upstream exactly."
    )

    def finalize(self, stage_2_handoff):
        dfr_layout = layout_from_handoff(stage_2_handoff)
        return (*finalize_stage2_result(stage_2_handoff, dfr_layout), float(stage_2_handoff.fps))


class LTXDFRSplitStage1AudioVideo:
    """Split a Stage-1 handoff into the two decode-ready latent streams."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_1_handoff": (
                    "LTX_DFR_STAGE2_HANDOFF",
                    {
                        "tooltip": (
                            "Single output from 'Run Stage 1 Spatial DFR'. It already contains the reserved "
                            "Stage-1 video and Stage-1 audio latents."
                        )
                    },
                ),
                "dfr_layout": (
                    "LTX_DFR_LAYOUT",
                    {"tooltip": "Used to trim the padded Stage-1 video back to the requested duration."},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("final_video_latent", "final_audio_latent")
    FUNCTION = "split"
    CATEGORY = "LTX/DFR Spatial/Stage 1"
    DESCRIPTION = (
        "Reads both streams directly from stage_1_handoff, trims the padded Stage-1 video to the requested "
        "duration, and returns the preserved Stage-1 audio latent. Decode the audio latent first, then use "
        "'LTX DFR: Trim Stage 1 Decoded Audio' for the exact waveform-length crop."
    )

    def split(self, stage_1_handoff, dfr_layout):
        return split_stage1_audio_video(stage_1_handoff, dfr_layout)


class LTXDFRValidateStage2FinalOutput:
    """Validate the packaged final Stage-2 outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": ("LTX_DFR_STAGE2_HANDOFF",),
                "stage_2_base_latent": ("LATENT",),
                "final_video_latent": ("LATENT",),
                "stage_1_audio_latent_for_decode": ("LATENT",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "video_trim_error",
        "audio_source_error",
        "video_shape_error",
        "video_framecount_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Checks that Stage 2I trimmed the padded Stage-2 base latent to exactly the requested latent/frame count and "
        "that the packaged final audio is exactly the Stage-1 audio preserved by the Stage-2 handoff."
    )

    def validate(self, stage_2_handoff, stage_2_base_latent, final_video_latent, stage_1_audio_latent_for_decode, dfr_layout):
        r = validate_stage2_final_output(
            stage_2_handoff=stage_2_handoff,
            stage_2_base_latent=stage_2_base_latent,
            final_video_latent=final_video_latent,
            final_audio_latent=stage_1_audio_latent_for_decode,
            dfr_layout=dfr_layout,
        )
        report = (
            f"PASS={r['passed']}; video_trim_error={r['video_trim_error']:.9g}; "
            f"audio_source_error={r['audio_source_error']:.9g}; "
            f"video_shape_error={r['final_video_requested_shape_error']:.9g}; "
            f"video_framecount_error={r['final_video_framecount_error']:.9g}; "
            f"requested_frames={r['requested_frames']}; padded_frames={r['padded_frames']}; "
            f"trimmed_pixel_frames={r['trimmed_pixel_frames']}; requested_latent_frames={r['requested_latent_frames']}; "
            f"padded_latent_frames={r['padded_latent_frames']}; trimmed_latent_frames={r['trimmed_latent_frames']}; "
            f"final_video_shape={r['final_video_shape']}; final_audio_shape={r['final_audio_shape']}; "
            f"duration_seconds={r['duration_seconds']:.9g}."
        )
        return {"ui": {"text": [report]}, "result": (
            r['passed'],
            r['video_trim_error'],
            r['audio_source_error'],
            r['final_video_requested_shape_error'],
            r['final_video_framecount_error'],
            report,
        )}


class LTXDFRTrimStage1DecodedAudio:
    """Trim decoded Stage-1 audio to the requested Stage-1 video duration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_1_handoff": (
                    "LTX_DFR_STAGE2_HANDOFF",
                    {"tooltip": "Single output from 'Run Stage 1 Spatial DFR'."},
                ),
                "decoded_audio": (
                    "AUDIO",
                    {"tooltip": "Audio decoded from the Stage-1 audio latent."},
                ),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("final_audio",)
    FUNCTION = "trim"
    CATEGORY = "LTX/DFR Spatial/Stage 1"
    DESCRIPTION = (
        "Crops decoded Stage-1 audio to round((requested_frames / fps) * sample_rate). "
        "Use after 'Split Stage 1 Audio / Video + Trim' and the Audio VAE decoder."
    )

    def trim(self, stage_1_handoff, decoded_audio, dfr_layout):
        return (trim_stage1_decoded_audio(stage_1_handoff, decoded_audio, dfr_layout),)


class LTXDFRTrimStage2DecodedAudio:
    """Trim the decoded preserved Stage-1 audio to the final requested video duration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": (
                    "LTX_DFR_STAGE2_RESULT_HANDOFF",
                    {
                        "tooltip": (
                            "Single output from 'Run Stage 2 AV Loop'. The final audio source remains Stage 1."
                        )
                    },
                ),
                "decoded_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Decode the final_audio_latent from 'Finalize Stage 2 Output / Trim' with LTXV Audio VAE Decode, "
                            "then connect that decoded AUDIO here. Upstream trims the waveform only after decoding."
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("final_audio",)
    FUNCTION = "trim"
    CATEGORY = "LTX/DFR Spatial/Stage 2 - Final Output"
    DESCRIPTION = (
        "Correct final Spatial-DFR audio epilogue. The official pipeline decodes the preserved Stage-1 audio latent first, "
        "then crops the decoded waveform to round((requested_frames / fps) * sample_rate). This node performs that exact "
        "post-decode trim so the muxed audio cannot outlast the trimmed final video."
    )

    def trim(self, stage_2_handoff, decoded_audio):
        dfr_layout = layout_from_handoff(stage_2_handoff)
        return (trim_stage2_result_decoded_audio(stage_2_handoff, decoded_audio, dfr_layout),)


class LTXDFRValidateStage2DecodedAudioTrim:
    """Validate the exact final decoded Stage-1 audio crop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": ("LTX_DFR_STAGE2_HANDOFF",),
                "decoded_stage_1_audio": ("AUDIO",),
                "final_audio": ("AUDIO",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "waveform_error",
        "sample_count_error",
        "sample_rate_error",
        "duration_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Independently reproduces upstream's final audio crop, requiring the returned AUDIO waveform to equal the exact "
        "prefix of the decoded Stage-1 audio, with the same sample rate and the expected final sample count."
    )

    def validate(self, stage_2_handoff, decoded_stage_1_audio, final_audio, dfr_layout):
        r = validate_stage2_decoded_audio_trim(
            stage_2_handoff=stage_2_handoff,
            decoded_stage_1_audio=decoded_stage_1_audio,
            final_audio=final_audio,
            dfr_layout=dfr_layout,
        )
        report = (
            f"PASS={r['passed']}; waveform_error={r['waveform_error']:.9g}; "
            f"sample_count_error={r['sample_count_error']:.9g}; sample_rate_error={r['sample_rate_error']:.9g}; "
            f"duration_error={r['duration_error']:.9g}; requested_frames={r['requested_frames']}; "
            f"playback_fps={r['playback_fps']:.9g}; video_seconds={r['video_seconds']:.9g}; "
            f"sample_rate={r['sample_rate']}; input_audio_samples={r['input_audio_samples']}; "
            f"expected_audio_samples={r['expected_audio_samples']}; final_audio_samples={r['final_audio_samples']}; "
            f"trimmed_audio_samples={r['trimmed_audio_samples']}; final_audio_seconds={r['final_audio_seconds']:.9g}; "
            f"final_waveform_shape={r['final_waveform_shape']}."
        )
        return {"ui": {"text": [report]}, "result": (
            r['passed'],
            r['waveform_error'],
            r['sample_count_error'],
            r['sample_rate_error'],
            r['duration_error'],
            report,
        )}


class LTXDFRMaterializeComfyModelInput:
    """Translate the noised official token state into current Comfy LTX model-call geometry."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"noised_official_state": ("LATENT",)}}

    RETURN_TYPES = ("LTX_DFR_MODEL_INPUT", "LATENT", "STRING")
    RETURN_NAMES = ("model_input", "materialized_latent", "report")
    FUNCTION = "materialize"
    CATEGORY = "LTX/DFR Spatial/Phase B - Execution Bridge"
    DESCRIPTION = (
        "Sub-step 2 of the strict DFR execution bridge. Converts the official patchified token state into "
        "the 5-D latent + denoise_mask + keyframe_idxs + generated_keyframes geometry consumed by Comfy's "
        "LTX model. It does not invoke the transformer."
    )

    def materialize(self, noised_official_state):
        model_input = materialize_comfy_model_input(noised_official_state)
        latent = {
            "samples": model_input.x.clone(),
            # Debug only: this mirrors the mask geometry that the eventual custom sampler/model call will pass.
            "noise_mask": model_input.denoise_mask.clone(),
        }
        gk = model_input.generated_keyframes
        report = (
            f"bridge_version={model_input.version}; materialized_shape={tuple(model_input.x.shape)}; "
            f"frame_rate={model_input.frame_rate:.9g}; base_frames={model_input.base_frames}; "
            f"appended_frames={model_input.appended_frames}; suffix_pre_filter={model_input.appended_tokens_before_filter}; "
            f"suffix_surviving={model_input.appended_tokens_after_filter}; total_tokens_after_filter={model_input.total_tokens_after_filter}; "
            f"keyframe_idxs_shape={None if model_input.keyframe_idxs is None else tuple(model_input.keyframe_idxs.shape)}; "
            f"generated_keyframes={gk}; operations={list(model_input.operation_frames)}."
        )
        return (model_input, latent, report)


class LTXDFRValidateComfyModelInput:
    """Round-trip validator for the Comfy LTX input adapter, before any model call."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_official_state": ("LATENT",),
                "model_input": ("LTX_DFR_MODEL_INPUT",),
                "sigmas": ("SIGMAS",),
                "step_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "max_latent_error",
        "max_mask_error",
        "max_position_error",
        "max_timestep_error",
        "max_marker_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    DESCRIPTION = (
        "Emulates Comfy LTXVModel._process_input and LTXV.process_timestep, then compares the surviving "
        "tokens, masks, positions, timesteps, and keyframe-marker semantics against the official noised token state."
    )
    OUTPUT_NODE = True

    def validate(self, noised_official_state, model_input, sigmas, step_index=0):
        if not isinstance(model_input, DFRComfyModelInput):
            raise ValueError(f"model_input has unexpected type {type(model_input)}.")
        if not torch.is_tensor(sigmas) or sigmas.ndim != 1:
            raise ValueError("sigmas must be a 1-D SIGMAS tensor.")
        step_index = int(step_index)
        if step_index < 0 or step_index >= sigmas.numel() - 1:
            raise ValueError(
                f"step_index={step_index} is outside the denoising transition range 0..{max(sigmas.numel()-2, 0)}."
            )
        sigma = float(sigmas[step_index].item())
        result = validate_model_input_roundtrip(noised_official_state, model_input, sigma=sigma)
        passed = bool(result["passed"])
        report = (
            f"PASS={passed}; sigma={sigma:.9g}; step_index={step_index}; "
            f"latent_max_abs_error={result['latent_error']:.9g}; mask_max_abs_error={result['mask_error']:.9g}; "
            f"position_max_abs_error={result['position_error']:.9g}; timestep_max_abs_error={result['timestep_error']:.9g}; "
            f"marker_max_abs_error={result['marker_error']:.9g}; generated_metadata_ok={result['generated_metadata_ok']}; "
            f"materialized_shape={result['materialized_shape']}; official_token_shape={result['official_token_shape']}; "
            f"suffix_pre_filter={result['suffix_pre_filter']}; suffix_surviving={result['suffix_surviving']}; "
            f"grid_filtered_tokens={result['grid_filtered_tokens']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                passed,
                float(result["latent_error"]),
                float(result["mask_error"]),
                float(result["position_error"]),
                float(result["timestep_error"]),
                float(result["marker_error"]),
                report,
            ),
        }


class LTXDFRVideoConditionByLatentIndex:
    """Port of Lightricks VideoConditionByLatentIndex conditioning semantics.

    This node accepts an already-encoded conditioning LATENT on purpose. Image
    preprocessing/VAE encoding is a separate Phase-A concern and must not be
    conflated with the conditioning operation we are reference-validating here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_latent": ("LATENT",),
                "conditioning_latent": ("LATENT",),
                "latent_idx": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999,
                        "step": 1,
                        "tooltip": (
                            "Official Lightricks latent index, not a decoded/pixel frame index. "
                            "For normal DFR I2V first-frame conditioning use 0."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Official semantics: writes 1-strength into the denoise mask for "
                            "the conditioned latent-frame token range."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("official_state_latent",)
    FUNCTION = "apply"
    CATEGORY = "LTX/DFR Spatial/Phase A - Official Conditioning"
    DESCRIPTION = (
        "Strict native-Comfy port of Lightricks VideoConditionByLatentIndex. "
        "It preserves the official separation between current latent, clean_latent, "
        "and denoise_mask in Phase-A metadata. It does not use LTXVAddGuide and does "
        "not yet materialize the state for stock KSampler; sampler/noiser parity is a "
        "later validation step."
    )

    def apply(self, target_latent, conditioning_latent, latent_idx=0, strength=1.0):
        return (
            apply_video_condition_by_latent_index(
                target_latent=target_latent,
                conditioning_latent=conditioning_latent,
                strength=strength,
                latent_idx=latent_idx,
            ),
        )


class LTXDFRVideoConditionByKeyframeIndex:
    """Port of Lightricks VideoConditionByKeyframeIndex conditioning semantics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_state_latent": ("LATENT",),
                "conditioning_latent": ("LATENT",),
                "frame_idx": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 999999,
                        "step": 1,
                        "tooltip": (
                            "Official pixel-frame index used for positional encoding. "
                            "For ordinary later keyframes this is the decoded video-frame index."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Official semantics: append guide tokens, keep zeros in the noisy latent, "
                            "and fill their denoise mask with 1-strength."
                        ),
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.001,
                        "tooltip": "Target output frame rate. Official positions divide the temporal axis by fps.",
                    },
                ),
                "num_pixel_frames": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 9999,
                        "step": 1,
                        "tooltip": (
                            "Official parameter. For a single still-image keyframe keep 1 so the temporal interval is [start, start+1)."
                        ),
                    },
                ),
                "bypass": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "When enabled, return the incoming official state unchanged and do not apply or "
                            "record this keyframe condition. The conditioning input's upstream branch may still run."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("official_state_latent",)
    FUNCTION = "apply"
    CATEGORY = "LTX/DFR Spatial/Phase A - Official Conditioning"
    DESCRIPTION = (
        "Strict native-Comfy port of Lightricks VideoConditionByKeyframeIndex. "
        "It appends guide tokens to the token sequence (zeros in the noisy latent, real tokens in clean_latent), "
        "computes official pixel-coordinate positions offset by frame_idx, and records the appended guide in "
        "the Phase-A token state."
    )

    def apply(
        self,
        official_state_latent,
        conditioning_latent,
        frame_idx=1,
        strength=1.0,
        fps=24.0,
        num_pixel_frames=1,
        bypass=False,
    ):
        if bool(bypass):
            return (official_state_latent,)
        return (
            apply_video_condition_by_keyframe_index(
                target_latent=official_state_latent,
                conditioning_latent=conditioning_latent,
                frame_idx=frame_idx,
                strength=strength,
                fps=fps,
                num_pixel_frames=num_pixel_frames,
            ),
        )


class LTXDFRVideoGeneratedKeyframeSlots:
    """Port of Lightricks VideoGeneratedKeyframeSlots semantics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_state_latent": ("LATENT",),
                "dfr_layout": (
                    "LTX_DFR_LAYOUT",
                    {
                        "tooltip": (
                            "Authoritative layout from Resolve DFR Canvas. Its official or explicitly custom "
                            "generated-slot positions are used unchanged."
                        )
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.001,
                        "tooltip": "Target output frame rate used for official slot positions.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("official_state_latent",)
    FUNCTION = "apply"
    CATEGORY = "LTX/DFR Spatial/Phase A - Official Conditioning"
    DESCRIPTION = (
        "Strict native-Comfy port of Lightricks VideoGeneratedKeyframeSlots. "
        "It appends extra generated slot tokens to the video token sequence, marks them as keyframe-class tokens, "
        "and records a generated_keyframe_layout so the denoised slots can later be extracted from the state."
    )

    def apply(self, official_state_latent, dfr_layout, fps=24.0):
        layout = validate_layout(dfr_layout)
        samples = official_state_latent.get("samples")
        if samples is None or not torch.is_tensor(samples) or samples.ndim != 5:
            raise ValueError("official_state_latent must contain [B,C,T,H,W] samples.")
        target_pixel_frames = (int(samples.shape[2]) - 1) * int(layout["temporal_scale"]) + 1
        if target_pixel_frames != int(layout["padded_frames"]):
            raise ValueError(
                "DFR layout/latent mismatch: layout requires padded_frames="
                f"{layout['padded_frames']}, but target latent represents {target_pixel_frames} pixel frames. "
                "Connect Resolve DFR Canvas's padded_video_length to EmptyLTXVLatentVideo.length."
            )
        out = apply_video_generated_keyframe_slots(
            target_latent=official_state_latent,
            pixel_frame_indices=list(layout["pixel_frame_indices"]),
            fps=fps,
        )
        out["dfr_layout"] = dict(layout)
        return (out,)


class LTXDFRVideoConditionByReferenceLatent:
    """Port of Lightricks VideoConditionByReferenceLatent conditioning semantics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_state_latent": ("LATENT",),
                "reference_latent": ("LATENT",),
                "downscale_factor": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Official target/reference spatial ratio. Spatial DFR reads this from the detailing IC-LoRA metadata; "
                            "for the current Pixel-Spatial x2 path this is normally 2."
                        ),
                    },
                ),
                "temporal_scale_factor": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": (
                            "Official target/reference temporal ratio. Spatial DFR uses 1; values >1 are supported for IC-LoRA parity."
                        ),
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Official semantics: appended reference denoise mask is 1-strength.",
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.001,
                        "tooltip": "Transformer conditioning fps used for the official reference RoPE positions.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("official_state_latent",)
    FUNCTION = "apply"
    CATEGORY = "LTX/DFR Spatial/Phase A - Official Conditioning"
    DESCRIPTION = (
        "Strict native-Comfy port of Lightricks VideoConditionByReferenceLatent. "
        "It appends the reference latent as clean IC-LoRA conditioning tokens with zero noisy placeholders, "
        "official temporal/spatial position translation, and no keyframe marker. Existing DFR generated-slot metadata is preserved."
    )

    def apply(
        self,
        official_state_latent,
        reference_latent,
        downscale_factor=2,
        temporal_scale_factor=1,
        strength=1.0,
        fps=24.0,
    ):
        return (
            apply_video_condition_by_reference_latent(
                target_latent=official_state_latent,
                reference_latent=reference_latent,
                downscale_factor=downscale_factor,
                temporal_scale_factor=temporal_scale_factor,
                strength=strength,
                fps=fps,
            ),
        )


class LTXDFRValidateReferenceLatent:
    """Diagnostic node for Phase-A reference-latent parity checks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_state_latent": ("LATENT",),
                "reference_latent": ("LATENT",),
                "downscale_factor": ("INT", {"default": 2, "min": 1, "max": 64, "step": 1}),
                "temporal_scale_factor": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 1000.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "max_clean_error",
        "max_mask_error",
        "max_zero_latent_error",
        "max_position_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase A - Tests"
    DESCRIPTION = (
        "Checks the appended VideoConditionByReferenceLatent tail: exact clean reference tokens, zero noisy placeholders, "
        "1-strength denoise mask, official positions, zero keyframe marker, and preservation of generated-keyframe layout."
    )
    OUTPUT_NODE = True

    def validate(
        self,
        official_state_latent,
        reference_latent,
        downscale_factor=2,
        temporal_scale_factor=1,
        strength=1.0,
        fps=24.0,
    ):
        state = get_official_state(official_state_latent)
        token_state = state.get("token_state")
        if token_state is None:
            raise ValueError("Official state has no token_state. Apply VideoConditionByReferenceLatent first.")
        ref = reference_latent.get("samples")
        if ref is None or not torch.is_tensor(ref) or ref.ndim != 5:
            raise ValueError("reference_latent must contain [B,C,F,H,W] samples.")

        ops = state.get("operations", [])
        if not ops or ops[-1].get("type") != "VideoConditionByReferenceLatent":
            raise ValueError("The last recorded official operation is not VideoConditionByReferenceLatent.")
        op = ops[-1]
        start = int(op["appended_token_start"])
        count = int(op["appended_token_count"])
        stop = start + count

        expected_tokens = ref.permute(0, 2, 3, 4, 1).reshape(ref.shape[0], -1, ref.shape[1]).to(
            device=token_state["clean_latent"].device,
            dtype=token_state["clean_latent"].dtype,
        )
        if expected_tokens.shape[1] != count:
            raise ValueError(
                f"Recorded appended_token_count={count} but reference latent patchifies to {expected_tokens.shape[1]} tokens."
            )

        clean_tail = token_state["clean_latent"][:, start:stop]
        latent_tail = token_state["latent"][:, start:stop]
        mask_tail = token_state["denoise_mask"][:, start:stop]
        pos_tail = token_state["positions"][:, :, start:stop]

        clean_err = float((clean_tail - expected_tokens).abs().max().item()) if clean_tail.numel() else 0.0
        zero_latent_err = float(latent_tail.abs().max().item()) if latent_tail.numel() else 0.0
        expected_mask = torch.full_like(mask_tail, 1.0 - float(strength))
        mask_err = float((mask_tail - expected_mask).abs().max().item()) if mask_tail.numel() else 0.0

        # Reproduce upstream position calculation exactly for the reference latent.
        b, _c, f, h, w = ref.shape
        grid_coords = torch.meshgrid(
            torch.arange(start=0, end=f, step=1, device=ref.device),
            torch.arange(start=0, end=h, step=1, device=ref.device),
            torch.arange(start=0, end=w, step=1, device=ref.device),
            indexing="ij",
        )
        starts = torch.stack(grid_coords, dim=0)
        ends = starts + 1
        coords = torch.stack((starts, ends), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1, 1, 1).reshape(b, 3, f * h * w, 2)
        scale_factors = tuple(int(x) for x in token_state.get("scale_factors", (8, 32, 32)))
        scale = torch.tensor(scale_factors, device=ref.device, dtype=coords.dtype).view(1, 3, 1, 1)
        expected_pos = coords * scale
        if bool(token_state.get("causal_fix", True)):
            expected_pos[:, 0, ...] = (expected_pos[:, 0, ...] + 1 - scale_factors[0]).clamp(min=0)
        expected_pos = expected_pos.to(dtype=torch.float32)
        expected_pos[:, 0, ...] /= float(fps) / int(temporal_scale_factor)
        if int(temporal_scale_factor) != 1:
            t_target = token_state["positions"][:, 0, 0:1, 1:2].to(device=expected_pos.device, dtype=torch.float32)
            expected_pos[:, 0, ...] = torch.clamp(
                expected_pos[:, 0, ...] - (int(temporal_scale_factor) - 1) * t_target,
                min=0,
            )
        if int(downscale_factor) != 1:
            expected_pos[:, 1, ...] *= int(downscale_factor)
            expected_pos[:, 2, ...] *= int(downscale_factor)
        expected_pos = expected_pos.to(device=pos_tail.device, dtype=torch.float32)
        position_err = float((pos_tail - expected_pos).abs().max().item()) if pos_tail.numel() else 0.0

        keyframes_mask_tail_err = 0.0
        if token_state.get("keyframes_mask") is not None:
            kf_tail = token_state["keyframes_mask"][:, start:stop]
            keyframes_mask_tail_err = float(kf_tail.abs().max().item()) if kf_tail.numel() else 0.0

        snapshot = op.get("generated_keyframe_layout_snapshot")
        current_layout = token_state.get("generated_keyframe_layout")
        layout_preserved = snapshot == current_layout
        attention_mask_ok = token_state.get("attention_mask") is None

        passed = (
            clean_err == 0.0
            and mask_err == 0.0
            and zero_latent_err == 0.0
            and position_err == 0.0
            and keyframes_mask_tail_err == 0.0
            and layout_preserved
            and attention_mask_ok
        )
        report = (
            f"PASS={passed}; clean_max_abs_error={clean_err:.9g}; mask_max_abs_error={mask_err:.9g}; "
            f"zero_latent_max_abs={zero_latent_err:.9g}; position_max_abs_error={position_err:.9g}; "
            f"keyframes_mask_tail_max_abs={keyframes_mask_tail_err:.9g}; generated_layout_preserved={layout_preserved}; "
            f"attention_mask_none={attention_mask_ok}; downscale_factor={int(downscale_factor)}; "
            f"temporal_scale_factor={int(temporal_scale_factor)}; strength={float(strength):.9g}; fps={float(fps):.9g}; "
            f"reference_tokens={count}; state_version={state.get('version')}; operations={len(ops)}."
        )
        return {
            "ui": {"text": [report]},
            "result": (passed, clean_err, mask_err, zero_latent_err, position_err, report),
        }


class LTXDFRValidateGeneratedKeyframeSlots:
    """Diagnostic node for Phase-A generated-keyframe-slot reference checks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_state_latent": ("LATENT",),
                "pixel_frame_indices": ("STRING", {"default": "24,48,72", "multiline": False}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 1000.0, "step": 0.001}),
            },
            "optional": {
                "initial_keyframes": ("LATENT",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "max_latent_error",
        "max_clean_error",
        "max_mask_error",
        "max_position_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase A - Tests"
    DESCRIPTION = (
        "Checks that VideoGeneratedKeyframeSlots appended the expected output tokens, keyframe mask, denoise mask, "
        "positions, and generated_keyframe_layout metadata."
    )
    OUTPUT_NODE = True

    def validate(self, official_state_latent, pixel_frame_indices, fps=24.0, initial_keyframes=None, dfr_layout=None):
        if dfr_layout is not None:
            indices = list(validate_layout(dfr_layout)["pixel_frame_indices"])
        else:
            indices = _parse_int_csv(pixel_frame_indices)
        state = get_official_state(official_state_latent)
        token_state = state.get("token_state")
        if token_state is None:
            raise ValueError("Official state has no token_state. Apply VideoGeneratedKeyframeSlots first.")
        layout = token_state.get("generated_keyframe_layout")
        if layout is None:
            raise ValueError("Official state has no generated_keyframe_layout.")
        ops = state.get("operations", [])
        if not ops or ops[-1].get("type") != "VideoGeneratedKeyframeSlots":
            raise ValueError("The last recorded official operation is not VideoGeneratedKeyframeSlots.")

        first = int(layout["first_token"])
        num_tokens = int(layout["num_tokens"])
        tpk = int(layout["tokens_per_keyframe"])
        num_keyframes = int(layout["num_keyframes"])
        stop = first + num_tokens
        if tuple(layout.get("pixel_frame_indices", ())) != tuple(indices):
            raise ValueError(
                f"Recorded pixel_frame_indices={layout.get('pixel_frame_indices')} do not match requested {tuple(indices)}."
            )
        if num_keyframes != len(indices):
            raise ValueError(f"Layout num_keyframes={num_keyframes} but requested {len(indices)} indices.")

        batch, channels, _t, height, width = state["base_shape"]
        if initial_keyframes is None:
            expected_frames = torch.zeros((batch, channels, num_keyframes, height, width), dtype=official_state_latent["samples"].dtype, device=official_state_latent["samples"].device)
        else:
            expected_frames = initial_keyframes.get("samples")
            if expected_frames is None or not torch.is_tensor(expected_frames) or expected_frames.ndim != 5:
                raise ValueError("initial_keyframes must contain [B,C,K,H,W] samples.")
            expected_frames = expected_frames.to(device=official_state_latent["samples"].device, dtype=official_state_latent["samples"].dtype)
        expected_tokens = expected_frames.permute(0, 2, 3, 4, 1).reshape(batch, num_keyframes * tpk, channels).to(
            device=token_state["latent"].device, dtype=token_state["latent"].dtype
        )

        latent_tail = token_state["latent"][:, first:stop]
        clean_tail = token_state["clean_latent"][:, first:stop]
        mask_tail = token_state["denoise_mask"][:, first:stop]
        pos_tail = token_state["positions"][:, :, first:stop]
        keyframes_mask_tail = token_state.get("keyframes_mask")
        if keyframes_mask_tail is None:
            raise ValueError("token_state.keyframes_mask is missing.")
        keyframes_mask_tail = keyframes_mask_tail[:, first:stop]

        latent_err = float((latent_tail - expected_tokens).abs().max().item()) if latent_tail.numel() else 0.0
        # Official generated slots always append zeros to clean_latent. Seed content lives only in latent.
        clean_err = float(clean_tail.abs().max().item()) if clean_tail.numel() else 0.0
        expected_mask = torch.ones_like(mask_tail)
        mask_err = float((mask_tail - expected_mask).abs().max().item()) if mask_tail.numel() else 0.0

        position_chunks = []
        scale_factors = tuple(int(x) for x in token_state.get("scale_factors", (8, 32, 32)))
        for pixel_idx in indices:
            grid_coords = torch.meshgrid(
                torch.arange(start=0, end=1, step=1, device=expected_frames.device),
                torch.arange(start=0, end=height, step=1, device=expected_frames.device),
                torch.arange(start=0, end=width, step=1, device=expected_frames.device),
                indexing="ij",
            )
            starts = torch.stack(grid_coords, dim=0)
            ends = starts + 1
            coords = torch.stack((starts, ends), dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1).reshape(batch, 3, height * width, 2)
            scale = torch.tensor(scale_factors, device=expected_frames.device, dtype=coords.dtype).view(1, 3, 1, 1)
            pos = coords * scale
            pos[:, 0, ...] += int(pixel_idx)
            pos[:, 0, ..., 1:] = pos[:, 0, ..., :1] + 1
            pos = pos.to(dtype=torch.float32, device=pos_tail.device)
            pos[:, 0, ...] = pos[:, 0, ...] / float(fps)
            position_chunks.append(pos)
        expected_pos = torch.cat(position_chunks, dim=2)
        position_err = float((pos_tail - expected_pos).abs().max().item()) if pos_tail.numel() else 0.0
        keyframes_mask_err = float((keyframes_mask_tail - torch.ones_like(keyframes_mask_tail)).abs().max().item()) if keyframes_mask_tail.numel() else 0.0

        extracted = extract_generated_keyframes(official_state_latent).to(device=expected_frames.device, dtype=expected_frames.dtype)
        extract_err = float((extracted - expected_frames).abs().max().item()) if extracted.numel() else 0.0

        passed = (
            latent_err == 0.0
            and clean_err == 0.0
            and mask_err == 0.0
            and position_err == 0.0
            and keyframes_mask_err == 0.0
            and extract_err == 0.0
        )
        report = (
            f"PASS={passed}; latent_max_abs_error={latent_err:.9g}; clean_max_abs_error={clean_err:.9g}; "
            f"mask_max_abs_error={mask_err:.9g}; position_max_abs_error={position_err:.9g}; "
            f"keyframes_mask_tail_max_abs={keyframes_mask_err:.9g}; extracted_keyframes_max_abs={extract_err:.9g}; "
            f"num_keyframes={num_keyframes}; appended_tokens={num_tokens}; base_token_count={int(token_state.get('base_token_count', -1))}; "
            f"state_version={state.get('version')}; operations={len(ops)}."
        )
        return {"ui": {"text": [report]}, "result": (passed, latent_err, clean_err, mask_err, position_err, report)}


class LTXDFRExtractGeneratedKeyframes:
    """Expose generated keyframes for visual/debug inspection only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"official_state_latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("generated_keyframes", "summary")
    FUNCTION = "extract"
    CATEGORY = "LTX/DFR Spatial/Phase A - Tests"
    DESCRIPTION = (
        "Debug helper. Extracts the generated_keyframes token slice as an unpatchified latent stack [B,C,K,H,W]."
    )

    def extract(self, official_state_latent):
        frames = extract_generated_keyframes(official_state_latent)
        state = get_official_state(official_state_latent)
        token_state = state.get("token_state")
        layout = token_state.get("generated_keyframe_layout") if token_state is not None else None
        summary = f"generated_keyframes={tuple(frames.shape)}, layout={layout}"
        return ({"samples": frames.clone()}, summary)


class LTXDFRValidateLatentIndex:
    """Diagnostic node for Phase-A reference checks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_state_latent": ("LATENT",),
                "conditioning_latent": ("LATENT",),
                "latent_idx": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "max_clean_error", "max_mask_error", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase A - Tests"
    DESCRIPTION = (
        "Checks that the Phase-A state contains the exact conditioning latent in clean_latent "
        "and exactly 1-strength in the corresponding denoise-mask range."
    )
    OUTPUT_NODE = True

    def validate(self, official_state_latent, conditioning_latent, latent_idx=0, strength=1.0):
        state = get_official_state(official_state_latent)
        cond = conditioning_latent.get("samples")
        if cond is None or not torch.is_tensor(cond) or cond.ndim != 5:
            raise ValueError("conditioning_latent must contain [B,C,T,H,W] samples.")

        latent_idx = int(latent_idx)
        strength = float(strength)
        stop = latent_idx + cond.shape[2]

        clean = state["clean_latent"][:, :, latent_idx:stop, :, :]
        mask = state["denoise_mask"][:, :, latent_idx:stop, :, :]
        cond_cmp = cond.to(device=clean.device, dtype=clean.dtype)

        if clean.shape != cond_cmp.shape:
            raise ValueError(
                f"Validation slice shape {tuple(clean.shape)} != conditioning shape {tuple(cond_cmp.shape)}."
            )

        clean_err = float((clean - cond_cmp).abs().max().item()) if clean.numel() else 0.0
        expected_mask = torch.full_like(mask, 1.0 - strength)
        mask_err = float((mask - expected_mask).abs().max().item()) if mask.numel() else 0.0
        passed = clean_err == 0.0 and mask_err == 0.0

        samples_unchanged = official_state_latent["samples"] is not state["clean_latent"]
        ops = state.get("operations", [])
        report = (
            f"PASS={passed}; clean_max_abs_error={clean_err:.9g}; "
            f"mask_max_abs_error={mask_err:.9g}; latent_idx={latent_idx}; "
            f"condition_latent_frames={cond.shape[2]}; expected_mask={1.0-strength:.9g}; "
            f"state_version={state.get('version')}; operations={len(ops)}; "
            f"current_latent_kept_separate_from_clean={samples_unchanged}."
        )
        return {
            "ui": {"text": [report]},
            "result": (passed, clean_err, mask_err, report),
        }


class LTXDFRValidateKeyframeIndex:
    """Diagnostic node for Phase-A keyframe-index reference checks."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_state_latent": ("LATENT",),
                "conditioning_latent": ("LATENT",),
                "frame_idx": ("INT", {"default": 1, "min": 0, "max": 999999, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 1000.0, "step": 0.001}),
                "num_pixel_frames": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "max_clean_error",
        "max_mask_error",
        "max_zero_latent_error",
        "max_position_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase A - Tests"
    DESCRIPTION = (
        "Checks that the official token-state contains an appended keyframe guide with: "
        "zero noisy-latent tokens, exact clean keyframe tokens, exact 1-strength mask values, "
        "and exact official positions for the appended guide tail."
    )
    OUTPUT_NODE = True

    def validate(self, official_state_latent, conditioning_latent, frame_idx=1, strength=1.0, fps=24.0, num_pixel_frames=1):
        state = get_official_state(official_state_latent)
        token_state = state.get("token_state")
        if token_state is None:
            raise ValueError("Official state has no token_state. Apply VideoConditionByKeyframeIndex first.")
        cond = conditioning_latent.get("samples")
        if cond is None or not torch.is_tensor(cond) or cond.ndim != 5:
            raise ValueError("conditioning_latent must contain [B,C,T,H,W] samples.")

        ops = state.get("operations", [])
        if not ops or ops[-1].get("type") != "VideoConditionByKeyframeIndex":
            raise ValueError("The last recorded official operation is not VideoConditionByKeyframeIndex.")
        op = ops[-1]
        start = int(op["appended_token_start"])
        count = int(op["appended_token_count"])
        stop = start + count

        expected_tokens = cond.permute(0, 2, 3, 4, 1).reshape(cond.shape[0], -1, cond.shape[1]).to(
            device=token_state["clean_latent"].device,
            dtype=token_state["clean_latent"].dtype,
        )
        if expected_tokens.shape[1] != count:
            raise ValueError(
                f"Recorded appended_token_count={count} but conditioning latent patchifies to {expected_tokens.shape[1]} tokens."
            )

        clean = token_state["clean_latent"][:, start:stop]
        latent_tail = token_state["latent"][:, start:stop]
        mask = token_state["denoise_mask"][:, start:stop]
        pos = token_state["positions"][:, :, start:stop]
        keyframe_mask_tail = token_state.get("keyframes_mask")
        if keyframe_mask_tail is not None:
            keyframe_mask_tail = keyframe_mask_tail[:, start:stop]

        clean_err = float((clean - expected_tokens).abs().max().item()) if clean.numel() else 0.0
        zero_latent_err = float(latent_tail.abs().max().item()) if latent_tail.numel() else 0.0
        expected_mask = torch.full_like(mask, 1.0 - float(strength))
        mask_err = float((mask - expected_mask).abs().max().item()) if mask.numel() else 0.0

        # Rebuild official expected positions.
        b, _, f, h, w = cond.shape
        grid_coords = torch.meshgrid(
            torch.arange(start=0, end=f, step=1, device=cond.device),
            torch.arange(start=0, end=h, step=1, device=cond.device),
            torch.arange(start=0, end=w, step=1, device=cond.device),
            indexing="ij",
        )
        starts = torch.stack(grid_coords, dim=0)
        ends = starts + 1
        coords = torch.stack((starts, ends), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1, 1, 1).reshape(b, 3, f * h * w, 2)
        scale = torch.tensor(token_state.get("scale_factors", (8, 32, 32)), device=cond.device, dtype=coords.dtype).view(1, 3, 1, 1)
        expected_pos = coords * scale
        if frame_idx == 0 and bool(token_state.get("causal_fix", True)):
            expected_pos[:, 0, ...] = (expected_pos[:, 0, ...] + 1 - scale[0, 0, 0, 0]).clamp(min=0)
        expected_pos[:, 0, ...] += int(frame_idx)
        if int(num_pixel_frames) == 1:
            expected_pos[:, 0, ..., 1:] = expected_pos[:, 0, ..., :1] + 1
        expected_pos = expected_pos.to(dtype=torch.float32, device=pos.device)
        expected_pos[:, 0, ...] = expected_pos[:, 0, ...] / float(fps)
        pos_err = float((pos - expected_pos).abs().max().item()) if pos.numel() else 0.0

        keyframe_mask_err = 0.0
        if keyframe_mask_tail is not None:
            keyframe_mask_err = float(keyframe_mask_tail.abs().max().item()) if keyframe_mask_tail.numel() else 0.0

        passed = (
            clean_err == 0.0
            and zero_latent_err == 0.0
            and mask_err == 0.0
            and pos_err == 0.0
            and keyframe_mask_err == 0.0
        )
        report = (
            f"PASS={passed}; clean_max_abs_error={clean_err:.9g}; mask_max_abs_error={mask_err:.9g}; "
            f"zero_latent_max_abs={zero_latent_err:.9g}; position_max_abs_error={pos_err:.9g}; "
            f"keyframes_mask_tail_max_abs={keyframe_mask_err:.9g}; frame_idx={int(frame_idx)}; "
            f"fps={float(fps):.9g}; num_pixel_frames={int(num_pixel_frames)}; appended_tokens={count}; "
            f"base_token_count={int(token_state.get('base_token_count', -1))}; state_version={state.get('version')}; operations={len(ops)}."
        )
        return {"ui": {"text": [report]}, "result": (passed, clean_err, mask_err, zero_latent_err, pos_err, report)}


class LTXDFRExecuteComfyDenoise:
    """Run one real Comfy/LTX denoise pass from the strict official DFR state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "noised_official_state": ("LATENT",),
                "sigmas": ("SIGMAS",),
                "step_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("raw_denoised_official_state", "report")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/Phase B - Layout & Execution"
    DESCRIPTION = (
        "Materializes the strict official token-state into native Comfy LTX model inputs, "
        "runs one real conditioned denoise pass through the provided MODEL, then repacks the returned "
        "5-D x0 prediction back into the strict official token ordering. This is the first actual transformer "
        "execution step of the parity path."
    )

    def run(self, model, positive, negative, noised_official_state, sigmas, step_index=0, cfg_scale=1.0, seed=42):
        denoised_official, _model_input, report = execute_comfy_model_denoise(
            model_patcher=model,
            positive=positive,
            negative=negative,
            noised_official_state=noised_official_state,
            sigmas=sigmas,
            step_index=step_index,
            cfg_scale=cfg_scale,
            seed=seed,
        )
        return (denoised_official, report)


class LTXDFRPostProcessDenoised:
    """Apply the official post_process_latent() blend in strict token space."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_official_state": ("LATENT",),
                "raw_denoised_official_state": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("post_processed_official_state", "report")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/Phase B - Layout & Execution"
    DESCRIPTION = (
        "Exact native-Comfy port of the official post_process_latent() step: blend the model x0 prediction with "
        "clean_latent using denoise_mask, while keeping the authoritative latent in the strict official token layout."
    )

    def run(self, noised_official_state, raw_denoised_official_state):
        post_processed, report = post_process_official_denoised(noised_official_state, raw_denoised_official_state)
        return (post_processed, report)


class LTXDFREulerStepFromDenoised:
    """Apply the official deterministic Euler update in strict token space."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_official_state": ("LATENT",),
                "raw_denoised_official_state": ("LATENT",),
                "sigmas": ("SIGMAS",),
                "step_index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("next_official_state", "post_processed_official_state", "report")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/Phase B - Layout & Execution"
    DESCRIPTION = (
        "Exact deterministic Euler sampler update using the strict official DFR token-state. The node first applies "
        "official post_process_latent() and then advances from sigma[step_index] to sigma[step_index+1]. When the "
        "next sigma is 0, it returns the post-processed denoised state directly, matching the official final-step logic."
    )

    def run(self, noised_official_state, raw_denoised_official_state, sigmas, step_index=0):
        next_state, post_processed, report = euler_step_from_official_denoised(
            noised_official_state=noised_official_state,
            raw_denoised_official_state=raw_denoised_official_state,
            sigmas=sigmas,
            step_index=step_index,
        )
        return (next_state, post_processed, report)


class LTXDFRValidateEulerStep:
    """Basic structural validator for the new execute/post-process/Euler chain."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noised_official_state": ("LATENT",),
                "raw_denoised_official_state": ("LATENT",),
                "next_official_state": ("LATENT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "raw_shape_error", "next_shape_error", "base_preview_error", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    DESCRIPTION = (
        "Structural sanity check for Phase B-05. Verifies that raw denoised and Euler-updated official states keep the "
        "same token shape as the input state, and that the updated base preview samples match the base-token slice."
    )

    def validate(self, noised_official_state, raw_denoised_official_state, next_official_state):
        noised = get_official_state(noised_official_state)
        raw = get_official_state(raw_denoised_official_state)
        nxt = get_official_state(next_official_state)
        noised_tokens = noised.get("token_state")
        raw_tokens = raw.get("token_state")
        next_tokens = nxt.get("token_state")
        if noised_tokens is None or raw_tokens is None or next_tokens is None:
            raise ValueError("All three LATENT inputs must already carry an official token_state.")

        raw_shape_error = 0.0 if tuple(raw_tokens["latent"].shape) == tuple(noised_tokens["latent"].shape) else 1.0
        next_shape_error = 0.0 if tuple(next_tokens["latent"].shape) == tuple(noised_tokens["latent"].shape) else 1.0

        base_shape = tuple(nxt["base_shape"])
        b, c, t, h, w = base_shape
        needed = t * h * w
        rebuilt = next_tokens["latent"][:, :needed].reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        preview = next_official_state["samples"].to(device=rebuilt.device, dtype=torch.float32)
        base_preview_error = float((rebuilt.to(torch.float32) - preview).abs().max().item())

        passed = raw_shape_error == 0.0 and next_shape_error == 0.0 and base_preview_error <= 1e-6
        report = (
            f"PASS={passed}; raw_shape_error={raw_shape_error:.9g}; next_shape_error={next_shape_error:.9g}; "
            f"base_preview_error={base_preview_error:.9g}; token_device={next_tokens['latent'].device}; "
            f"preview_source_device={next_official_state['samples'].device}; "
            f"token_shape={tuple(int(x) for x in noised_tokens['latent'].shape)}; base_shape={base_shape}."
        )
        return {"ui": {"text": [report]}, "result": (passed, raw_shape_error, next_shape_error, base_preview_error, report)}


class LTXDFRRunStage1DFRLoop:
    """Run all eight official deterministic Stage-1 Spatial-DFR denoising transitions."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "noised_official_state": ("LATENT",),
                "stage_1_sigmas": ("SIGMAS",),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Keep 1.0 for the distilled official-parity Stage-1 test.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "STRING")
    RETURN_NAMES = ("final_official_state", "stage_1_base_latent", "generated_keyframes", "report")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Spatial/Phase B - Layout & Execution"
    DESCRIPTION = (
        "Runs the complete official Spatial-DFR Stage-1 deterministic Euler trajectory using the exact 8-step "
        "distilled sigma schedule. The Comfy LTX model is prepared once and reused across all eight transformer "
        "evaluations. Returns the final strict official state, the ordinary Stage-1 base video latent, and the "
        "real model-generated DFR slot keyframes needed by Stage 2."
    )

    def run(self, model, positive, negative, noised_official_state, stage_1_sigmas, cfg_scale=1.0, seed=42):
        return execute_stage1_denoising_loop(
            model_patcher=model,
            positive=positive,
            negative=negative,
            noised_official_state=noised_official_state,
            sigmas=stage_1_sigmas,
            cfg_scale=cfg_scale,
            seed=seed,
        )


class LTXDFRValidateStage1DFRLoop:
    """Verify that Stage-1 exported base/slot latents are exact slices of the final official state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "final_official_state": ("LATENT",),
                "stage_1_base_latent": ("LATENT",),
                "generated_keyframes": ("LATENT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "base_max_error", "generated_keyframes_max_error", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/Phase B - Tests"
    DESCRIPTION = (
        "Checks that the Stage-1 base latent and generated-keyframe stack returned by the full loop are exact "
        "unpatchified views of the final strict official token state."
    )
    OUTPUT_NODE = True

    def validate(self, final_official_state, stage_1_base_latent, generated_keyframes):
        passed, base_err, kf_err, report = validate_stage1_loop_outputs(
            final_official_state,
            stage_1_base_latent,
            generated_keyframes,
        )
        return {"ui": {"text": [report]}, "result": (passed, base_err, kf_err, report)}


class LTXDFRExtractCleanLatent:
    """Expose official clean_latent for visual/debug inspection only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"official_state_latent": ("LATENT",)}}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("clean_latent", "summary")
    FUNCTION = "extract"
    CATEGORY = "LTX/DFR Spatial/Phase A - Tests"
    DESCRIPTION = (
        "Debug helper. Extracts the official clean_latent carried in Phase-A metadata. "
        "Do not feed this debug output to the final DFR sampler path."
    )

    def extract(self, official_state_latent):
        state = get_official_state(official_state_latent)
        clean = state["clean_latent"]
        mask = state["denoise_mask"]
        out = {"samples": clean.clone()}
        summary = (
            f"clean_latent={tuple(clean.shape)}, denoise_mask={tuple(mask.shape)}, "
            f"mask_min={float(mask.min().item()):.6g}, mask_max={float(mask.max().item()):.6g}, "
            f"state_key={OFFICIAL_STATE_KEY}"
        )
        return (out, summary)


class LTXDFRConditioningResizeCenterCrop:
    """Official LTX DFR geometric conditioning resize: bilinear fill-resize + centered crop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {
                        "default": 768,
                        "min": 1,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Target output width in pixels.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Target output height in pixels.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "scale", "resized_width", "resized_height", "report")
    FUNCTION = "resize"
    CATEGORY = "LTX/DFR Spatial/Phase C - Conditioning"
    DESCRIPTION = (
        "Official LTX DFR geometric conditioning resize: aspect-preserving fill resize with bilinear "
        "interpolation, align_corners=False, ceil-rounded resized dimensions, and centered crop to the "
        "exact target size. Use this after LTXVPreprocess when reproducing official conditioning prep."
    )

    def resize(self, image, width=768, height=512):
        src_h = int(image.shape[1])
        src_w = int(image.shape[2])
        scale = max(float(height) / float(src_h), float(width) / float(src_w))
        resized_h = int(__import__('math').ceil(float(src_h) * scale))
        resized_w = int(__import__('math').ceil(float(src_w) * scale))
        out = bilinear_fill_resize_center_crop(image, height, width)
        report = (
            f"source={src_w}x{src_h}; target={width}x{height}; scale={scale:.8f}; "
            f"resized={resized_w}x{resized_h}; mode=bilinear; align_corners=False; "
            "crop=center."
        )
        return (out, float(scale), resized_w, resized_h, report)


class LTXDFRProbeFinalDecoderRuntimeU33A:
    """U3.3a: inspect the actual loaded Comfy DiffVAE interface and restore decoder RNG."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("rng_state_restored", "generator_path_found", "report")
    FUNCTION = "probe"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "U3.3a preflight. Performs no decode. Restores the exact post-Stage2 shared RNG state from U3.1, "
        "then introspects the concrete loaded Comfy VAE and first-stage decoder methods/signatures so U3.3b can "
        "target the real generator-capable decoder interface instead of guessing a private API."
    )
    OUTPUT_NODE = True

    def probe(self, vae, decoder_handoff):
        probe, _details = probe_decoder_runtime(vae, decoder_handoff)
        report = runtime_probe_report(probe)
        return {
            "ui": {"text": [report]},
            "result": (
                bool(probe.generator_state_restored),
                bool(probe.candidate_accepts_generator),
                report,
            ),
        }


class LTXDFRProbeDecoderKeyframeRuntimeU34A:
    """U3.4a: inspect whether the loaded decoder runtime exposes any plausible keyframe-aware path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
                "final_decode_keyframes": ("LTX_DFR_DECODE_KEYFRAMES",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("rng_state_restored", "final_keyframes_present", "runtime_support_found", "report")
    FUNCTION = "probe"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "U3.4a preflight. Performs no decode. Validates the U3.2 final decoder-keyframe package, "
        "restores the exact post-Stage2 RNG state from U3.1, and inspects the loaded Comfy/LTX decoder "
        "tree for plausible keyframe-aware entry points (keyframes parameters, forward_with_keyframes helpers, "
        "and type-embedding receivers)."
    )
    OUTPUT_NODE = True

    def probe(self, vae, decoder_handoff, final_decode_keyframes):
        probe, _details = probe_decoder_keyframe_runtime(vae, decoder_handoff, final_decode_keyframes)
        report = keyframe_runtime_probe_report(probe)
        return {
            "ui": {"text": [report]},
            "result": (
                bool(probe.generator_state_restored),
                bool(probe.keyframes_present),
                bool(keyframe_runtime_supported(probe)),
                report,
            ),
        }


class LTXDFRPrepareDecoderKeyframeSubstrateU34B1:
    """U3.4b1: build the structural keyframe-decoder substrate object."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
                "final_decode_keyframes": ("LTX_DFR_DECODE_KEYFRAMES",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_DECODER_KEYFRAME_SUBSTRATE", "STRING")
    RETURN_NAMES = ("decoder_keyframe_substrate", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "U3.4b1 preflight. Performs no decode. Re-validates the U3.2 final decoder-keyframe package, "
        "restores the exact post-Stage2 shared RNG state from U3.1, computes the canonical latent-border "
        "positions of the final decoder keyframes, and inventories the loaded decoder modules/parameters that "
        "later U3.4b stages will need (type_emb, conv_in, conv_in_x_t, timestep receivers, upsamplers, "
        "generic blocks, output head)."
    )
    OUTPUT_NODE = True

    def prepare(self, vae, decoder_handoff, final_decode_keyframes, dfr_layout):
        substrate, report = build_decoder_keyframe_substrate(vae, decoder_handoff, final_decode_keyframes, dfr_layout)
        return {
            "ui": {"text": [report]},
            "result": (substrate, report),
        }


class LTXDFRValidateDecoderKeyframeSubstrateU34B1:
    """U3.4b1 validation node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
                "final_decode_keyframes": ("LTX_DFR_DECODE_KEYFRAMES",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
                "decoder_keyframe_substrate": ("LTX_DFR_DECODER_KEYFRAME_SUBSTRATE",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("passed", "type_emb_present", "channels_match_expected", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "Rebuilds the U3.4b1 decoder-keyframe substrate from the connected runtime and compares every structural "
        "field against the provided substrate object. This still performs no decode."
    )
    OUTPUT_NODE = True

    def validate(self, vae, decoder_handoff, final_decode_keyframes, dfr_layout, decoder_keyframe_substrate):
        result = validate_decoder_keyframe_substrate(
            vae,
            decoder_handoff,
            final_decode_keyframes,
            dfr_layout,
            decoder_keyframe_substrate,
        )
        report = (
            f"PASS={result['passed']}; stage=U3.4b1_validate; metadata_error={result['metadata_error']}; "
            f"type_emb_present={result['has_type_emb']}; has_conv_in={result['has_conv_in']}; "
            f"has_conv_in_x_t={result['has_conv_in_x_t']}; channels_match_expected={result['channels_match_expected']}; "
            f"keyframe_count={result['keyframe_count']}; keyframe_latent_indices={result['keyframe_latent_indices']}; "
            f"type_emb_paths={result['type_emb_paths']}; conv_in_paths={result['conv_in_paths']}; "
            f"conv_in_x_t_paths={result['conv_in_x_t_paths']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                bool(result['passed']),
                bool(result['has_type_emb']),
                bool(result['channels_match_expected']),
                report,
            ),
        }


class LTXDFRExtractDecoderKeyframeCheckpointWeightsU34B1B:
    """U3.4b1b: inspect the raw VAE checkpoint and recover decoder.type_emb."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            choices = folder_paths.get_filename_list("vae")
        except Exception:
            choices = []
        return {
            "required": {
                "decoder_keyframe_substrate": ("LTX_DFR_DECODER_KEYFRAME_SUBSTRATE",),
                "vae_name": (choices,),
            }
        }

    RETURN_TYPES = ("LTX_DFR_DECODER_KEYFRAME_CHECKPOINT_WEIGHTS", "BOOLEAN", "STRING")
    RETURN_NAMES = ("decoder_keyframe_checkpoint_weights", "checkpoint_has_type_emb", "report")
    FUNCTION = "extract"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "U3.4b1b raw-checkpoint preflight. Reads the selected VAE safetensors directly, checks for the official "
        "decoder type_emb tensor, validates its 128-channel width, and packages the exact tensor for later "
        "keyframe-aware decoder stages. If and only if the checkpoint truly lacks type_emb, it records the "
        "official zero fallback. This node does not mutate the loaded VAE and performs no decode."
    )
    OUTPUT_NODE = True

    def extract(self, decoder_keyframe_substrate, vae_name):
        package, report = extract_decoder_keyframe_checkpoint_weights(decoder_keyframe_substrate, vae_name)
        return {
            "ui": {"text": [report]},
            "result": (package, bool(package.checkpoint_has_type_emb), report),
        }


class LTXDFRValidateDecoderKeyframeCheckpointWeightsU34B1B:
    """Validate exact raw-checkpoint extraction for U3.4b1b."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths
            choices = folder_paths.get_filename_list("vae")
        except Exception:
            choices = []
        return {
            "required": {
                "decoder_keyframe_substrate": ("LTX_DFR_DECODER_KEYFRAME_SUBSTRATE",),
                "vae_name": (choices,),
                "decoder_keyframe_checkpoint_weights": ("LTX_DFR_DECODER_KEYFRAME_CHECKPOINT_WEIGHTS",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("passed", "checkpoint_has_type_emb", "zero_fallback", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    OUTPUT_NODE = True

    def validate(self, decoder_keyframe_substrate, vae_name, decoder_keyframe_checkpoint_weights):
        result = validate_decoder_keyframe_checkpoint_weights(
            decoder_keyframe_substrate, vae_name, decoder_keyframe_checkpoint_weights
        )
        report = (
            f"PASS={result['passed']}; stage=U3.4b1b_validate; metadata_error={result['metadata_error']}; "
            f"tensor_error={result['tensor_error']}; checkpoint_has_type_emb={result['checkpoint_has_type_emb']}; "
            f"source_key={result['source_key']}; type_emb_shape={result['type_emb_shape']}; "
            f"type_emb_dtype={result['type_emb_dtype']}; zero_fallback={result['zero_fallback']}; "
            f"channels_match_expected={result['channels_match_expected']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                bool(result['passed']),
                bool(result['checkpoint_has_type_emb']),
                bool(result['zero_fallback']),
                report,
            ),
        }


class LTXDFRPrepareJointDecoderAttentionU34B2:
    """U3.4b2: prepare the isolated official joint video/keyframe attention port."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decoder_keyframe_substrate": ("LTX_DFR_DECODER_KEYFRAME_SUBSTRATE",),
                "decoder_keyframe_checkpoint_weights": ("LTX_DFR_DECODER_KEYFRAME_CHECKPOINT_WEIGHTS",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_JOINT_DECODER_ATTENTION", "STRING")
    RETURN_NAMES = ("joint_decoder_attention", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "U3.4b2 mathematical port. Verifies the validated U3.4b1 substrate and the real trained U3.4b1b "
        "decoder.type_emb, then exposes the official auto joint video/keyframe neighborhood-attention path: "
        "Triton on CUDA when available and eager otherwise. It does not mutate or run the VAE decoder yet."
    )
    OUTPUT_NODE = True

    def prepare(self, decoder_keyframe_substrate, decoder_keyframe_checkpoint_weights):
        state, report = prepare_joint_decoder_attention(
            decoder_keyframe_substrate, decoder_keyframe_checkpoint_weights
        )
        return {"ui": {"text": [report]}, "result": (state, report)}


class LTXDFRValidateJointDecoderAttentionU34B2:
    """U3.4b2 semantic validator against a brute-force attention oracle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "joint_decoder_attention": ("LTX_DFR_JOINT_DECODER_ATTENTION",),
                "test_device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "video_max_error", "keyframe_max_error", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "Runs the U3.4b2 joint-attention port on small deterministic tensors and compares both video and "
        "keyframe streams against an independent brute-force oracle. Also validates invalid-plane masking, "
        "cross-stream influence, stable nearest-slot tie-breaking, and far-keyframe visibility independent of Kt."
    )
    OUTPUT_NODE = True

    def validate(self, joint_decoder_attention, test_device="cuda"):
        result = validate_joint_decoder_attention(joint_decoder_attention, test_device)
        report = (
            f"PASS={result['passed']}; stage=U3.4b2_validate_joint_attention; device={result['device']}; "
            f"runtime_backend={result['runtime_backend']}; "
            f"tolerance={result['tolerance']:.9g}; video_max_error={result['video_error']:.9g}; "
            f"keyframe_max_error={result['keyframe_error']:.9g}; "
            f"invalid_video_error={result['invalid_video_error']:.9g}; "
            f"invalid_keyframe_error={result['invalid_keyframe_error']:.9g}; "
            f"invalid_plane_zero_error={result['invalid_plane_zero_error']:.9g}; "
            f"shape_error={result['shape_error']:.9g}; finite_error={result['finite_error']:.9g}; "
            f"cross_stream_influence={result['influence']:.9g}; influence_error={result['influence_error']:.9g}; "
            f"slot_tie_error={result['slot_tie_error']:.9g}; far_visibility_error={result['far_visibility_error']:.9g}; "
            f"video_shape={result['video_shape']}; keyframe_shape={result['keyframe_shape']}; "
            "decoder_mutated=False; decoder_called=False."
        )
        return {
            "ui": {"text": [report]},
            "result": (bool(result['passed']), float(result['video_error']), float(result['keyframe_error']), report),
        }


class LTXDFRDecodeFinalVideoWithContinuedRNGU33B:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",),
            "final_video_latent": ("LATENT",),
            "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
            "tile_x": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
            "tile_y": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
            "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
            "tile_t": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4}),
            "overlap_t": ("INT", {"default": 16, "min": 4, "max": 4096, "step": 4}),
        }}
    RETURN_TYPES = ("IMAGE", "LTX_DFR_DECODER_BRIDGE_STATE", "STRING")
    RETURN_NAMES = ("images", "decoder_bridge_state", "report")
    FUNCTION = "decode"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    OUTPUT_NODE = True
    DESCRIPTION = ("U3.3b corrected bridge: preserves core VAEDecodeTiled geometry and latent un-normalization, "
                   "but replaces CausalDiffusionVAE's per-call seed-0 generator with the continued Stage2 RNG. "
                   "Keyframe anchoring remains disabled until U3.4.")
    def decode(self, vae, final_video_latent, decoder_handoff, tile_x=512, tile_y=512, overlap=64, tile_t=64, overlap_t=16):
        images, state, report = decode_final_video_with_continued_rng(
            vae, final_video_latent, decoder_handoff, tile_x, tile_y, overlap, tile_t, overlap_t)
        return {"ui": {"text": [report]}, "result": (images, state, report)}


class LTXDFRValidateFinalDecoderBridgeU33B:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
            "decoder_bridge_state": ("LTX_DFR_DECODER_BRIDGE_STATE",),
        }}
    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "initial_rng_error", "rng_advanced_error", "keyframes_error", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    OUTPUT_NODE = True
    def validate(self, decoder_handoff, decoder_bridge_state):
        r = validate_decoder_bridge(decoder_handoff, decoder_bridge_state)
        report = (f"PASS={r['passed']}; stage=U3.3b_validate; initial_rng_error={r['initial_rng_error']:.9g}; "
                  f"rng_advanced_error={r['rng_advanced_error']:.9g}; keyframes_error={r['keyframes_error']:.9g}; "
                  f"tile_calls_error={r['tile_calls_error']:.9g}; path_error={r['path_error']:.9g}; "
                  f"output_format_error={r['output_format_error']:.9g}; tile_decode_calls={r['tile_decode_calls']}; "
                  f"pixel_tiles={r['pixel_tiles']}; latent_tiles={r['latent_tiles']}; "
                  f"wrapper_output_shape={r['wrapper_output_shape']}; output_shape={r['output_shape']}.")
        return {"ui": {"text": [report]}, "result": (r['passed'], r['initial_rng_error'], r['rng_advanced_error'], r['keyframes_error'], report)}


class LTXDFRPrepareDeterministicDecoderKeyframeContextU34B3A:
    """U3.4b3a: prepare the deterministic decoder stage/keyframe context."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "decoder_keyframe_substrate": ("LTX_DFR_DECODER_KEYFRAME_SUBSTRATE",),
                "decoder_keyframe_checkpoint_weights": ("LTX_DFR_DECODER_KEYFRAME_CHECKPOINT_WEIGHTS",),
                "joint_decoder_attention": ("LTX_DFR_JOINT_DECODER_ATTENTION",),
                "final_decode_keyframes": ("LTX_DFR_DECODE_KEYFRAMES",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_DETERMINISTIC_DECODER_KEYFRAME_CONTEXT", "STRING")
    RETURN_NAMES = ("deterministic_decoder_keyframe_context", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "U3.4b3a safe split. Performs no decoder execution yet. It combines the validated U3.4b1 substrate, "
        "the exact U3.4b1b checkpoint type_emb, the validated U3.4b2 joint-attention package, and the U3.2 "
        "final decoder-keyframe package into the deterministic stage/keyframe context that later U3.4b3b will "
        "feed through the real decoder stages 1-4."
    )
    OUTPUT_NODE = True

    def prepare(self, vae, decoder_keyframe_substrate, decoder_keyframe_checkpoint_weights, joint_decoder_attention, final_decode_keyframes, dfr_layout):
        context, report = prepare_deterministic_decoder_keyframe_context(
            vae,
            decoder_keyframe_substrate,
            decoder_keyframe_checkpoint_weights,
            joint_decoder_attention,
            final_decode_keyframes,
            dfr_layout,
        )
        return {
            "ui": {"text": [report]},
            "result": (context, report),
        }


class LTXDFRValidateDeterministicDecoderKeyframeContextU34B3A:
    """U3.4b3a validator."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "decoder_keyframe_substrate": ("LTX_DFR_DECODER_KEYFRAME_SUBSTRATE",),
                "decoder_keyframe_checkpoint_weights": ("LTX_DFR_DECODER_KEYFRAME_CHECKPOINT_WEIGHTS",),
                "joint_decoder_attention": ("LTX_DFR_JOINT_DECODER_ATTENTION",),
                "final_decode_keyframes": ("LTX_DFR_DECODE_KEYFRAMES",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
                "deterministic_decoder_keyframe_context": ("LTX_DFR_DETERMINISTIC_DECODER_KEYFRAME_CONTEXT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("passed", "geometry_and_order_valid", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "Rebuilds the U3.4b3a deterministic stage/keyframe context and compares it against the provided object. "
        "Still performs no decoder execution."
    )
    OUTPUT_NODE = True

    def validate(self, vae, decoder_keyframe_substrate, decoder_keyframe_checkpoint_weights, joint_decoder_attention, final_decode_keyframes, dfr_layout, deterministic_decoder_keyframe_context):
        result = validate_deterministic_decoder_keyframe_context(
            vae,
            decoder_keyframe_substrate,
            decoder_keyframe_checkpoint_weights,
            joint_decoder_attention,
            final_decode_keyframes,
            dfr_layout,
            deterministic_decoder_keyframe_context,
        )
        report = (
            f"PASS={result['passed']}; stage=U3.4b3a_geometry_fixed_validate; metadata_error={result['metadata_error']}; "
            f"tensor_error={result['tensor_error']}; unnormalize_then_type_emb_error={result['unnormalize_then_type_emb_error']}; "
            f"stage_count={result['stage_count']}; deterministic_stage_count={result['deterministic_stage_count']}; "
            f"upsample_strides={result['upsample_strides']}; temporal_scale_schedule={result['temporal_scale_schedule']}; "
            f"spatial_scale_schedule={result['spatial_scale_schedule']}; "
            f"deterministic_output_spatial_shape={result['deterministic_output_spatial_shape']}; "
            f"final_pixel_spatial_shape={result['final_pixel_spatial_shape']}; "
            f"stage_names={result['stage_names']}; final_stage_keyframe_indices={result['final_stage_keyframe_indices']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (bool(result['passed']), bool(result['unnormalize_then_type_emb_error'] == 0.0), report),
        }


class LTXDFRPrepareDeterministicStageOfficialPreflightU34B3B1:
    """C23b: official-source-defined preflight before deterministic dual-stream port."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "deterministic_decoder_keyframe_context": ("LTX_DFR_DETERMINISTIC_DECODER_KEYFRAME_CONTEXT",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_OFFICIAL_DETERMINISTIC_STAGE_PREFLIGHT", "STRING")
    RETURN_NAMES = ("official_deterministic_stage_preflight", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "C23b replacement for C23. Uses the frozen official DiffusionVideoDecoder/keyframes algorithm as the contract. "
        "Runtime inspection only verifies how Comfy exposes the checkpoint-derived backbone. The keyframe upsample probe "
        "uses the exact official upsample_keyframe_planes rule and never treats the keyframe planes as an ordinary video stream."
    )
    OUTPUT_NODE = True

    def prepare(self, vae, deterministic_decoder_keyframe_context):
        state, report = prepare_official_deterministic_stage_preflight(
            vae, deterministic_decoder_keyframe_context
        )
        return {"ui": {"text": [report]}, "result": (state, report)}


class LTXDFRValidateDeterministicStageOfficialPreflightU34B3B1:
    """Validator for C23b official-source preflight."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "deterministic_decoder_keyframe_context": ("LTX_DFR_DETERMINISTIC_DECODER_KEYFRAME_CONTEXT",),
                "official_deterministic_stage_preflight": ("LTX_DFR_OFFICIAL_DETERMINISTIC_STAGE_PREFLIGHT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("passed", "official_mapping_valid", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "Rebuilds the frozen-source-defined C23b preflight. PASS requires exact checkpoint-config/runtime backbone mapping, "
        "exact official keyframe-plane upsample semantics, invariant plane count, real checkpoint type_emb, and the expected "
        "absence of native forward_with_keyframes methods in current Comfy."
    )
    OUTPUT_NODE = True

    def validate(self, vae, deterministic_decoder_keyframe_context, official_deterministic_stage_preflight):
        r = validate_official_deterministic_stage_preflight(
            vae,
            deterministic_decoder_keyframe_context,
            official_deterministic_stage_preflight,
        )
        report = (
            f"PASS={r['passed']}; stage=U3.4b3b1_official_source_validate; metadata_error={r['metadata_error']}; "
            f"architecture_matches_checkpoint_config={r['architecture_matches_checkpoint_config']}; "
            f"keyframe_plane_count_invariant={r['keyframe_plane_count_invariant']}; "
            f"keyframe_plane_helper_exact={r['keyframe_plane_helper_exact']}; "
            f"remaining_time_strides={r['remaining_time_strides']}; keyframe_stage_times={r['keyframe_stage_times']}; "
            f"checkpoint_stage_channels={r['checkpoint_stage_channels']}; runtime_stage_channels={r['runtime_stage_channels']}; "
            f"checkpoint_stage_depths={r['checkpoint_stage_depths']}; runtime_stage_depths={r['runtime_stage_depths']}; "
            f"checkpoint_stage_kernels={r['checkpoint_stage_kernels']}; runtime_stage_kernels={r['runtime_stage_kernels']}; "
            f"checkpoint_upsamples={r['checkpoint_upsamples']}; runtime_upsample_strides={r['runtime_upsample_strides']}; "
            f"runtime_upsample_reductions={r['runtime_upsample_reductions']}; "
            f"missing_keyframe_methods_expected={r['missing_keyframe_methods_expected']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                bool(r["passed"]),
                bool(r["architecture_matches_checkpoint_config"] and r["keyframe_plane_helper_exact"]),
                report,
            ),
        }


class LTXDFRPrepareDeterministicDualStreamBlockPortU34B3B2:
    """Prepare the frozen-source-defined deterministic NABlock dual-stream bridge."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "official_deterministic_stage_preflight": ("LTX_DFR_OFFICIAL_DETERMINISTIC_STAGE_PREFLIGHT",),
            "joint_decoder_attention": ("LTX_DFR_JOINT_DECODER_ATTENTION",),
        }}
    RETURN_TYPES = ("LTX_DFR_DETERMINISTIC_DUAL_STREAM_BLOCK_PORT", "STRING")
    RETURN_NAMES = ("deterministic_dual_stream_block_port", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "U3.4b3b2. Declares the external no-mutation port of the frozen official NABlock.forward_with_keyframes behavior. "
        "Requires C23b and the validated U3.4b2 joint attention."
    )
    OUTPUT_NODE = True
    def prepare(self, official_deterministic_stage_preflight, joint_decoder_attention):
        state, report = prepare_deterministic_dual_stream_block_port(
            official_deterministic_stage_preflight, joint_decoder_attention
        )
        return {"ui":{"text":[report]}, "result":(state, report)}


class LTXDFRValidateDeterministicDualStreamBlockPortU34B3B2:
    """Execute one real Comfy NABlock through the port and validate single-stream reduction."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae": ("VAE",),
            "deterministic_dual_stream_block_port": ("LTX_DFR_DETERMINISTIC_DUAL_STREAM_BLOCK_PORT",),
            "stage_index": ("INT", {"default":3,"min":0,"max":3,"step":1}),
            "block_index": ("INT", {"default":0,"min":0,"max":32,"step":1}),
        }}
    RETURN_TYPES = ("BOOLEAN","FLOAT","FLOAT","STRING")
    RETURN_NAMES = ("passed","native_video_max_error","cross_stream_influence","report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "U3.4b3b2 runtime validator. With all keyframe planes invalid, the ported video branch must reduce to the loaded "
        "Comfy NABlock's native single-stream output within floating-point tolerance. With valid planes it additionally proves "
        "real cross-stream influence. The decoder is never mutated."
    )
    OUTPUT_NODE = True
    def validate(self, vae, deterministic_dual_stream_block_port, stage_index=3, block_index=0):
        r=validate_deterministic_dual_stream_block_port(
            vae, deterministic_dual_stream_block_port, stage_index, block_index
        )
        report=(
            f"PASS={r['passed']}; stage=U3.4b3b2_validate_dual_stream_block; device={r['device']}; dtype={r['dtype']}; "
            f"stage_index={r['stage_index']}; block_index={r['block_index']}; kernel={r['kernel']}; channels={r['channels']}; "
            f"video_shape={r['video_shape']}; keyframe_shape={r['keyframe_shape']}; "
            f"native_video_max_error={r['native_video_max_error']:.9g}; tolerance={r['tolerance']:.9g}; "
            f"invalid_keyframe_zero_error={r['invalid_keyframe_zero_error']:.9g}; "
            f"cross_stream_influence={r['cross_stream_influence']:.9g}; finite_error={r['finite_error']:.9g}; "
            f"shape_error={r['shape_error']:.9g}; weight_mutation_error={r['weight_mutation_error']:.9g}; decoder_mutated=False."
        )
        return {"ui":{"text":[report]}, "result":(bool(r['passed']),float(r['native_video_max_error']),float(r['cross_stream_influence']),report)}


class LTXDFRPrepareExactDeterministicDualStreamPortC24B:
    """C24b: source-exact deterministic dual-stream semantics on the Comfy decoder backbone."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_deterministic_stage_preflight": ("LTX_DFR_OFFICIAL_DETERMINISTIC_STAGE_PREFLIGHT",),
                "joint_decoder_attention": ("LTX_DFR_JOINT_DECODER_ATTENTION",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_EXACT_DETERMINISTIC_DUAL_STREAM_PORT", "STRING")
    RETURN_NAMES = ("exact_deterministic_dual_stream_port", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "C24b replacement for failed C24. Binds the frozen official NABlock/NeighborhoodAttention/"
        "_run_det_stage_with_keyframes semantics to the existing loaded Comfy NADiffusionDecoder weights. "
        "Invalid keyframe planes are NOT re-zeroed inside a block; they are masked only after the stage upsample."
    )
    OUTPUT_NODE = True

    def prepare(self, official_deterministic_stage_preflight, joint_decoder_attention):
        state, report = prepare_exact_deterministic_dual_stream_port(
            official_deterministic_stage_preflight,
            joint_decoder_attention,
        )
        return {"ui": {"text": [report]}, "result": (state, report)}


class LTXDFRExecuteExactDeterministicStagesC24B:
    """C24b: execute the official deterministic dual-stream path on real Stage-2 data."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "final_video_latent": ("LATENT",),
                "deterministic_decoder_keyframe_context": ("LTX_DFR_DETERMINISTIC_DECODER_KEYFRAME_CONTEXT",),
                "exact_deterministic_dual_stream_port": ("LTX_DFR_EXACT_DETERMINISTIC_DUAL_STREAM_PORT",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_DETERMINISTIC_STAGE_EXECUTION", "STRING")
    RETURN_NAMES = ("deterministic_stage_execution", "report")
    FUNCTION = "execute"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "Executes frozen-source-exact deterministic stages 1-3 on the real final Stage-2 video latent and generated "
        "decoder keyframes using the existing Comfy VAE/ModelPatcher lifecycle. It also executes the exact stage-4 "
        "dual-stream hop on a real origin feature crop. Full stage 4 is intentionally left coupled to Stage 5, as in "
        "the official tiled decoder, rather than materializing a multi-GB full context volume. No Stage-5 block runs here."
    )
    OUTPUT_NODE = True

    def execute(self, vae, final_video_latent, deterministic_decoder_keyframe_context, exact_deterministic_dual_stream_port):
        state, report = execute_exact_deterministic_stages(
            vae,
            final_video_latent,
            deterministic_decoder_keyframe_context,
            exact_deterministic_dual_stream_port,
        )
        return {"ui": {"text": [report]}, "result": (state, report)}


class LTXDFRValidateExactDeterministicStagesC24B:
    """C24b validator for source-defined deterministic execution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "deterministic_stage_execution": ("LTX_DFR_DETERMINISTIC_STAGE_EXECUTION",),
                "deterministic_decoder_keyframe_context": ("LTX_DFR_DETERMINISTIC_DECODER_KEYFRAME_CONTEXT",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "cross_stream_influence", "invalid_keyframe_zero_error", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "Validates C24b only against frozen-source invariants: exact stage geometry, fractional keyframe times, "
        "constant plane count, stage-boundary invalid-plane masking, real cross-stream influence, unchanged decoder "
        "weights/RNG, and no Stage-5 execution. It deliberately does NOT compare against native Comfy NATTEN at borders."
    )
    OUTPUT_NODE = True

    def validate(self, deterministic_stage_execution, deterministic_decoder_keyframe_context):
        r = validate_exact_deterministic_stages(
            deterministic_stage_execution,
            deterministic_decoder_keyframe_context,
        )
        report = (
            f"PASS={r['passed']}; stage=U3.4b3_C24b_validate_exact_det; "
            f"geometry_error={r['geometry_error']}; timing_error={r['timing_error']}; "
            f"plane_count_error={r['plane_count_error']}; valid_count_error={r['valid_count_error']}; "
            f"stage4_time_error={r['stage4_time_error']}; stage4_geometry_error={r['stage4_geometry_error']}; "
            f"stage4_probe_cross_stream_influence={r['stage4_probe_cross_stream_influence']:.9g}; "
            f"stage4_probe_invalid_keyframe_zero_error={r['stage4_probe_invalid_keyframe_zero_error']:.9g}; "
            f"decoder_parameters_unchanged={r['decoder_parameters_unchanged']}; "
            f"cpu_rng_unchanged={r['cpu_rng_unchanged']}; cuda_rng_unchanged={r['cuda_rng_unchanged']}; "
            f"stage5_called={r['stage5_called']}; stage_video_shapes={r['stage_video_shapes']}; "
            f"stage_keyframe_shapes={r['stage_keyframe_shapes']}; stage_keyframe_times={r['stage_keyframe_times']}; "
            f"stage4_probe_video_output_shape={r['stage4_probe_video_output_shape']}; "
            f"stage4_probe_keyframe_output_shape={r['stage4_probe_keyframe_output_shape']}; "
            f"stage4_probe_output_times={r['stage4_probe_output_times']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                bool(r["passed"]),
                float(r["stage4_probe_cross_stream_influence"]),
                float(r["stage4_probe_invalid_keyframe_zero_error"]),
                report,
            ),
        }


class LTXDFRPrepareExactStage5DualStreamPortC25:
    """C25: bind frozen combined Stage-5 dual-stream semantics to Comfy weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "exact_deterministic_dual_stream_port": ("LTX_DFR_EXACT_DETERMINISTIC_DUAL_STREAM_PORT",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_EXACT_STAGE5_DUAL_STREAM_PORT", "STRING")
    RETURN_NAMES = ("exact_stage5_dual_stream_port", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "C25 / U3.4b4. Declares the frozen official combined Stage-5 dual-stream contract on the existing loaded "
        "Comfy NADiffusionDecoder weights. Uses independent shared context projection, shared AdaLN, exact keyframe "
        "RoPE times, validated joint attention, shared SwiGLU, and Stage-5 per-block invalid-plane re-zeroing."
    )
    OUTPUT_NODE = True

    def prepare(self, exact_deterministic_dual_stream_port):
        state, report = prepare_exact_stage5_dual_stream_port(exact_deterministic_dual_stream_port)
        return {"ui": {"text": [report]}, "result": (state, report)}


class LTXDFRExecuteExactStage5SingleTileC25:
    """C25: execute Stage 4 + all eight Stage-5 dual-stream blocks on one real origin tile."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "deterministic_stage_execution": ("LTX_DFR_DETERMINISTIC_STAGE_EXECUTION",),
                "deterministic_decoder_keyframe_context": ("LTX_DFR_DETERMINISTIC_DECODER_KEYFRAME_CONTEXT",),
                "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
                "exact_stage5_dual_stream_port": ("LTX_DFR_EXACT_STAGE5_DUAL_STREAM_PORT",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_STAGE5_SINGLE_TILE_EXECUTION", "STRING")
    RETURN_NAMES = ("stage5_single_tile_execution", "report")
    FUNCTION = "execute"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "Executes the frozen official combined Stage-5 keyframe path on a source-safe real origin crop from C24b. "
        "Restores the exact post-Stage2 AV generator, draws video pixel noise then separate keyframe-pixel noise in "
        "official order, runs real Stage 4 and every Stage-5 dual-stream block, and returns the validation state. "
        "This is deliberately one tile only; C26b adds the frozen official tile schedule, per-tile plane selection, current-checkpoint per-tile RNG ordering, halos and blending."
    )
    OUTPUT_NODE = True

    def execute(
        self,
        vae,
        deterministic_stage_execution,
        deterministic_decoder_keyframe_context,
        decoder_handoff,
        exact_stage5_dual_stream_port,
    ):
        state, report = execute_exact_stage5_single_tile(
            vae,
            deterministic_stage_execution,
            deterministic_decoder_keyframe_context,
            decoder_handoff,
            exact_stage5_dual_stream_port,
        )
        return {"ui": {"text": [report]}, "result": (state, report)}


class LTXDFRValidateExactStage5SingleTileC25:
    """C25 validator against frozen Stage-5 source invariants only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage5_single_tile_execution": ("LTX_DFR_STAGE5_SINGLE_TILE_EXECUTION",),
                "deterministic_decoder_keyframe_context": ("LTX_DFR_DETERMINISTIC_DECODER_KEYFRAME_CONTEXT",),
                "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("passed", "stage5_cross_stream_influence", "invalid_keyframe_zero_error", "report")
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "Validates C25 against the frozen combined Stage-5 contract: exact Stage-4/Stage-5 geometry and times, "
        "one-step x0 checkpoint contract, all eight blocks executed, real Stage-5 keyframe influence, per-block "
        "invalid-plane zeroing, continued RNG restoration/advance, unchanged global RNG and unchanged decoder weights."
    )
    OUTPUT_NODE = True

    def validate(self, stage5_single_tile_execution, deterministic_decoder_keyframe_context, decoder_handoff):
        r = validate_exact_stage5_single_tile(
            stage5_single_tile_execution,
            deterministic_decoder_keyframe_context,
            decoder_handoff,
        )
        report = (
            f"PASS={r['passed']}; stage=U3.4b4_C25_validate_exact_stage5_single_tile; "
            f"geometry_error={r['geometry_error']}; keyframe_geometry_error={r['keyframe_geometry_error']}; "
            f"timing_error={r['timing_error']}; checkpoint_error={r['checkpoint_error']}; "
            f"output_geometry_error={r['output_geometry_error']}; initial_rng_error={r['initial_rng_error']}; "
            f"rng_advance_error={r['rng_advance_error']}; stage5_cross_stream_influence={r['stage5_cross_stream_influence']:.9g}; "
            f"invalid_keyframe_zero_error={r['invalid_keyframe_zero_error']:.9g}; finite_error={r['finite_error']:.9g}; "
            f"generator_state_restored={r['generator_state_restored']}; generator_state_advanced={r['generator_state_advanced']}; "
            f"cpu_rng_unchanged={r['cpu_rng_unchanged']}; cuda_rng_unchanged={r['cuda_rng_unchanged']}; "
            f"decoder_parameters_unchanged={r['decoder_parameters_unchanged']}; stage5_called={r['stage5_called']}; "
            f"stage5_all_blocks_called={r['stage5_all_blocks_called']}; stage5_block_count={r['stage5_block_count']}; "
            f"stage5_blocks_executed={r['stage5_blocks_executed']}; "
            f"stage4_input_tile_shape={r['stage4_input_tile_shape']}; stage4_context_shape={r['stage4_context_shape']}; "
            f"stage5_keyframe_context_shape={r['stage5_keyframe_context_shape']}; stage5_keyframe_times={r['stage5_keyframe_times']}; "
            f"pixel_noise_shape={r['pixel_noise_shape']}; keyframe_pixel_noise_shape={r['keyframe_pixel_noise_shape']}; "
            f"video_prediction_shape={r['video_prediction_shape']}; keyframe_prediction_shape={r['keyframe_prediction_shape']}; "
            f"plane_selection_mode={r['plane_selection_mode']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                bool(r["passed"]),
                float(r["stage5_cross_stream_influence"]),
                float(r["invalid_keyframe_zero_error"]),
                report,
            ),
        }


class LTXDFRPrepareAutoFullDecodeScheduleC26B:
    """C26b auto path: select an official memory-aware schedule for this run."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "deterministic_stage_execution": ("LTX_DFR_DETERMINISTIC_STAGE_EXECUTION",),
                "deterministic_decoder_keyframe_context": ("LTX_DFR_DETERMINISTIC_DECODER_KEYFRAME_CONTEXT",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_C26B_TILED_SCHEDULE", "STRING")
    RETURN_NAMES = ("official_tiled_decode_schedule", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "Automatically selects the final DiffVAE tile sizes from the loaded decoder geometry and current free VRAM. "
        "It ports the official LTX memory-aware recommender for the validated keyframe-aware CHUNKED_EAGER/Triton path, "
        "then emits the same schedule type accepted by Execute Exact Full Decode. The separate manual schedule node is unchanged."
    )
    OUTPUT_NODE = True

    def prepare(
        self,
        vae,
        deterministic_stage_execution,
        deterministic_decoder_keyframe_context,
    ):
        state, report = prepare_official_auto_tiled_decode_schedule(
            vae,
            deterministic_stage_execution,
            deterministic_decoder_keyframe_context,
        )
        return {"ui": {"text": [report]}, "result": (state, report)}


class LTXDFRPrepareExactFullDecodeScheduleC26B:
    """C26b: build the exact frozen DiffVAE tile schedule for the final keyframe decode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "deterministic_stage_execution": ("LTX_DFR_DETERMINISTIC_STAGE_EXECUTION",),
                "deterministic_decoder_keyframe_context": ("LTX_DFR_DETERMINISTIC_DECODER_KEYFRAME_CONTEXT",),
                "tile_frames": ("INT", {"default": 104, "min": 1, "max": 10000, "step": 1}),
                "temporal_overlap": ("INT", {"default": 40, "min": 0, "max": 9999, "step": 1}),
                "tile_height": ("INT", {"default": 416, "min": 1, "max": 16384, "step": 1}),
                "tile_width": ("INT", {"default": 544, "min": 1, "max": 16384, "step": 1}),
                "spatial_overlap": ("INT", {"default": 160, "min": 0, "max": 16383, "step": 1}),
            }
        }

    RETURN_TYPES = ("LTX_DFR_C26B_TILED_SCHEDULE", "STRING")
    RETURN_NAMES = ("official_tiled_decode_schedule", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "C26b / U3.4b5. Builds the final decoder tile schedule from the frozen LTX DiffVAE tiling source, "
        "using the loaded decoder's exact Stage-4/Stage-5 kernels, depths, last upsample stride and patch size. "
        "The defaults reproduce the official current-target schedule: 104/40 frames, 416/160 height and "
        "544/160 width, yielding 12 tiles in one temporal group at 97x896x1664."
    )
    OUTPUT_NODE = True

    def prepare(
        self,
        vae,
        deterministic_stage_execution,
        deterministic_decoder_keyframe_context,
        tile_frames=104,
        temporal_overlap=40,
        tile_height=416,
        tile_width=544,
        spatial_overlap=160,
    ):
        state, report = prepare_official_tiled_decode_schedule(
            vae,
            deterministic_stage_execution,
            deterministic_decoder_keyframe_context,
            tile_frames=int(tile_frames),
            temporal_overlap=int(temporal_overlap),
            tile_height=int(tile_height),
            tile_width=int(tile_width),
            spatial_overlap=int(spatial_overlap),
        )
        return {"ui": {"text": [report]}, "result": (state, report)}


class LTXDFRExecuteExactFullDecodeC26B:
    """C26b: execute the complete frozen tiled keyframe-aware final DiffVAE decode."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "deterministic_stage_execution": ("LTX_DFR_DETERMINISTIC_STAGE_EXECUTION",),
                "deterministic_decoder_keyframe_context": ("LTX_DFR_DETERMINISTIC_DECODER_KEYFRAME_CONTEXT",),
                "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
                "exact_stage5_dual_stream_port": ("LTX_DFR_EXACT_STAGE5_DUAL_STREAM_PORT",),
                "official_tiled_decode_schedule": ("LTX_DFR_C26B_TILED_SCHEDULE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LTX_DFR_C26B_TILED_EXECUTION", "STRING")
    RETURN_NAMES = ("decoded_images", "official_tiled_decode_execution", "report")
    FUNCTION = "execute"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "C26b / U3.4b5. Executes the exact frozen tiled final keyframe decode on the validated C24b/C25 neural path: "
        "official tile geometry and masks, per-tile keyframe planes, separate Stage-4/pixel origins, ghost handling, "
        "current one-step-x0 per-tile video-noise then keyframe-noise RNG order, temporal overlap handoff and final pixel assembly."
    )
    OUTPUT_NODE = True

    def execute(
        self,
        vae,
        deterministic_stage_execution,
        deterministic_decoder_keyframe_context,
        decoder_handoff,
        exact_stage5_dual_stream_port,
        official_tiled_decode_schedule,
    ):
        image, state, report = execute_official_tiled_decode(
            vae,
            deterministic_stage_execution,
            deterministic_decoder_keyframe_context,
            decoder_handoff,
            exact_stage5_dual_stream_port,
            official_tiled_decode_schedule,
        )
        return {"ui": {"text": [report]}, "result": (image, state, report)}


class LTXDFRValidateExactFullDecodeC26B:
    """C26b validator for the exact frozen full tiled decoder orchestration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "official_tiled_decode_execution": ("LTX_DFR_C26B_TILED_EXECUTION",),
                "official_tiled_decode_schedule": ("LTX_DFR_C26B_TILED_SCHEDULE",),
                "decoder_handoff": ("LTX_DFR_DECODER_HANDOFF",),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "passed",
        "geometry_error",
        "tile_error",
        "plane_error",
        "rng_error",
        "invariants_error",
        "report",
    )
    FUNCTION = "validate"
    CATEGORY = "LTX/DFR Spatial/U3 - Final Decoder"
    DESCRIPTION = (
        "Validates C26b output geometry, exact tile/group/block counts, exact per-tile plane sets, continued generator "
        "restoration/advance, global RNG isolation, and decoder non-mutation."
    )
    OUTPUT_NODE = True

    def validate(self, official_tiled_decode_execution, official_tiled_decode_schedule, decoder_handoff):
        r = validate_official_tiled_decode(
            official_tiled_decode_execution, official_tiled_decode_schedule, decoder_handoff
        )
        rng_error = max(float(r["rng_initial_error"]), float(r["rng_advance_error"]))
        invariants_error = max(
            float(r["invariants_error"]), float(r["group_error"]), float(r["block_error"]),
        )
        report = (
            f"PASS={r['passed']}; stage=U3.4b5_C26b_validate_official_tiled_decode; "
            f"geometry_error={r['geometry_error']}; tile_error={r['tile_error']}; group_error={r['group_error']}; "
            f"block_error={r['block_error']}; plane_error={r['plane_error']}; "
            f"rng_initial_error={r['rng_initial_error']}; rng_advance_error={r['rng_advance_error']}; "
            f"invariants_error={r['invariants_error']}; "
            f"output_image={r['output_image_shape']}; expected_image={r['expected_image_shape']}; "
            f"tiles={r['rendered_tiles']}/{r['total_tiles']}; groups={r['rendered_groups']}/{r['temporal_groups']}; "
            f"stage5_blocks={r['stage5_blocks_executed_total']}/{r['expected_stage5_block_calls']}; "
            f"planes_per_tile={r['min_planes_per_tile']}..{r['max_planes_per_tile']}; "
            f"generator_state_restored={r['generator_state_restored']}; generator_state_advanced={r['generator_state_advanced']}; "
            f"cpu_rng_unchanged={r['cpu_rng_unchanged']}; cuda_rng_unchanged={r['cuda_rng_unchanged']}; "
            f"decoder_parameters_unchanged={r['decoder_parameters_unchanged']}; "
            f"complementary_masks={r['complementary_masks']}; group_slices={r['group_slices']}."
        )
        return {
            "ui": {"text": [report]},
            "result": (
                bool(r["passed"]),
                float(r["geometry_error"]),
                float(r["tile_error"]),
                float(r["plane_error"]),
                float(rng_error),
                float(invariants_error),
                report,
            ),
        }


# Only the completed/user-facing nodes are registered with ComfyUI.
NODE_CLASS_MAPPINGS = {
    'LTXDFRExperimentalBlockStreamingDecode_DFRSpatial': LTXDFRExperimentalBlockStreamingDecode,
    'LTXDFRResolveDFRCanvas_DFRSpatial': LTXDFRResolveDFRCanvas,
    'LTXDFRConditioningResizeCenterCrop_DFRSpatial': LTXDFRConditioningResizeCenterCrop,
    'LTXDFRValidateDFRModelCapability_DFRSpatial': LTXDFRValidateDFRModelCapability,
    'LTXDFRSigmaSchedules_DFRSpatial': LTXDFRSigmaSchedules,
    'LTXDFRCreateStage1AudioState_DFRSpatial': LTXDFRCreateStage1AudioState,
    'LTXDFRVideoConditionByLatentIndex_DFRSpatial': LTXDFRVideoConditionByLatentIndex,
    'LTXDFRVideoConditionByKeyframeIndex_DFRSpatial': LTXDFRVideoConditionByKeyframeIndex,
    'LTXDFRVideoGeneratedKeyframeSlots_DFRSpatial': LTXDFRVideoGeneratedKeyframeSlots,
    'LTXDFRRunStage1SpatialDFR_DFRSpatial': LTXDFRRunStage1SpatialDFR,
    'LTXDFRSplitStage1AudioVideo_DFRSpatial': LTXDFRSplitStage1AudioVideo,
    'LTXSpatialDFRStage1PreviewDecode_DFRSpatial': LTXSpatialDFRStage1PreviewDecode,
    'LTXDFRTrimStage1DecodedAudio_DFRSpatial': LTXDFRTrimStage1DecodedAudio,
    'LTXDFRRunStage2SpatialUpscale_DFRSpatial': LTXDFRRunStage2SpatialUpscale,
    'LTXDFRRunModularStage2SpatialUpscale_DFRSpatial': LTXDFRRunModularStage2SpatialUpscale,
    'LTXDFRRunStage2SpatialUpscaleFromStage2HandoffTest_DFRSpatial': LTXDFRRunStage2SpatialUpscaleFromStage2HandoffTest,
    'LTXDFRPrepareStage2DetailingModel_DFRSpatial': LTXDFRPrepareStage2DetailingModel,
    'LTXDFRFinalizeStage2DFRConditioning_DFRSpatial': LTXDFRFinalizeStage2DFRConditioning,
    'LTXDFRRunStage2AVLoop_DFRSpatial': LTXDFRRunStage2AVLoop,
    'LTXDFRFinalizeStage2Output_DFRSpatial': LTXDFRFinalizeStage2Output,
    'LTXDFRSpatialDFRVideoDecode_DFRSpatial': LTXDFRSpatialDFRVideoDecode,
    'LTXDFRTrimStage2DecodedAudio_DFRSpatial': LTXDFRTrimStage2DecodedAudio,
}

# Development, validation, experimental, and atomic construction nodes are kept
# in source for regression/debug work but intentionally NOT registered with ComfyUI.
DEV_NODE_CLASS_MAPPINGS = {
    'LTXDFRPrepareOfficialFinalDecode_DFRSpatial': LTXDFRPrepareOfficialFinalDecode,
    'LTXDFRPrepareStage2DecoderHandoff_DFRSpatial': LTXDFRPrepareStage2DecoderHandoff,
    'LTXDFRBuildFinalDecodeKeyframes_DFRSpatial': LTXDFRBuildFinalDecodeKeyframes,
    'LTXDFRValidateFinalDecodeKeyframes_DFRSpatial': LTXDFRValidateFinalDecodeKeyframes,
    'LTXDFRValidateStage2DecoderHandoff_DFRSpatial': LTXDFRValidateStage2DecoderHandoff,
    'LTXExperimentalResolveDenseSlotCanvas16_DFRSpatial': LTXExperimentalResolveDenseSlotCanvas16,
    'LTXDFRValidateDFRCanvas_DFRSpatial': LTXDFRValidateDFRCanvas,
    'LTXDFRValidateDFRSigmaSchedules_DFRSpatial': LTXDFRValidateDFRSigmaSchedules,
    'LTXDFRGaussianNoiser_DFRSpatial': LTXDFRGaussianNoiser,
    'LTXDFRValidateGaussianNoiser_DFRSpatial': LTXDFRValidateGaussianNoiser,
    'LTXDFRValidateStage1AudioState_DFRSpatial': LTXDFRValidateStage1AudioState,
    'LTXDFRAudioGaussianNoiser_DFRSpatial': LTXDFRAudioGaussianNoiser,
    'LTXDFRValidateAudioGaussianNoiser_DFRSpatial': LTXDFRValidateAudioGaussianNoiser,
    'LTXDFRStage1AVGaussianNoiser_DFRSpatial': LTXDFRStage1AVGaussianNoiser,
    'LTXDFRValidateStage1AVGaussianNoiser_DFRSpatial': LTXDFRValidateStage1AVGaussianNoiser,
    'LTXDFRMaterializeStage1AVModelInput_DFRSpatial': LTXDFRMaterializeStage1AVModelInput,
    'LTXDFRValidateStage1AVModelInput_DFRSpatial': LTXDFRValidateStage1AVModelInput,
    'LTXDFRExecuteStage1AVDenoise_DFRSpatial': LTXDFRExecuteStage1AVDenoise,
    'LTXDFRValidateStage1AVDenoise_DFRSpatial': LTXDFRValidateStage1AVDenoise,
    'LTXDFRStage1AVEulerStepFromDenoised_DFRSpatial': LTXDFRStage1AVEulerStepFromDenoised,
    'LTXDFRValidateStage1AVEulerStep_DFRSpatial': LTXDFRValidateStage1AVEulerStep,
    'LTXDFRRunStage1AVLoop_DFRSpatial': LTXDFRRunStage1AVLoop,
    'LTXDFRValidateStage1AVLoop_DFRSpatial': LTXDFRValidateStage1AVLoop,
    'LTXDFRPrepareStage2Handoff_DFRSpatial': LTXDFRPrepareStage2Handoff,
    'LTXDFRValidateStage2Handoff_DFRSpatial': LTXDFRValidateStage2Handoff,
    'LTXDFRValidateStage2SpatialUpscale_DFRSpatial': LTXDFRValidateStage2SpatialUpscale,
    'LTXDFRResolveStage2DetailingMetadata_DFRSpatial': LTXDFRResolveStage2DetailingMetadata,
    'LTXDFRValidateStage2DFRConditioning_DFRSpatial': LTXDFRValidateStage2DFRConditioning,
    'LTXDFRApplyStage2DetailingICLoRA_DFRSpatial': LTXDFRApplyStage2DetailingICLoRA,
    'LTXDFRValidateStage2DetailingICLoRA_DFRSpatial': LTXDFRValidateStage2DetailingICLoRA,
    'LTXDFRStage2AVGaussianNoiser_DFRSpatial': LTXDFRStage2AVGaussianNoiser,
    'LTXDFRValidateStage2AVGaussianNoiser_DFRSpatial': LTXDFRValidateStage2AVGaussianNoiser,
    'LTXDFRExecuteStage2AVDenoise_DFRSpatial': LTXDFRExecuteStage2AVDenoise,
    'LTXDFRValidateStage2AVDenoise_DFRSpatial': LTXDFRValidateStage2AVDenoise,
    'LTXDFRStage2AVEulerStepFromDenoised_DFRSpatial': LTXDFRStage2AVEulerStepFromDenoised,
    'LTXDFRValidateStage2AVEulerStep_DFRSpatial': LTXDFRValidateStage2AVEulerStep,
    'LTXExperimentalRunStage2AVLoopExtraStep_DFRSpatial': LTXExperimentalRunStage2AVLoopExtraStep,
    'LTXDFRValidateStage2AVLoop_DFRSpatial': LTXDFRValidateStage2AVLoop,
    'LTXDFRValidateStage2FinalOutput_DFRSpatial': LTXDFRValidateStage2FinalOutput,
    'LTXDFRValidateStage2DecodedAudioTrim_DFRSpatial': LTXDFRValidateStage2DecodedAudioTrim,
    'LTXDFRMaterializeComfyModelInput_DFRSpatial': LTXDFRMaterializeComfyModelInput,
    'LTXDFRValidateComfyModelInput_DFRSpatial': LTXDFRValidateComfyModelInput,
    'LTXDFRExecuteComfyDenoise_DFRSpatial': LTXDFRExecuteComfyDenoise,
    'LTXDFRPostProcessDenoised_DFRSpatial': LTXDFRPostProcessDenoised,
    'LTXDFREulerStepFromDenoised_DFRSpatial': LTXDFREulerStepFromDenoised,
    'LTXDFRValidateEulerStep_DFRSpatial': LTXDFRValidateEulerStep,
    'LTXDFRRunStage1DFRLoop_DFRSpatial': LTXDFRRunStage1DFRLoop,
    'LTXDFRValidateStage1DFRLoop_DFRSpatial': LTXDFRValidateStage1DFRLoop,
    'LTXDFRVideoConditionByReferenceLatent_DFRSpatial': LTXDFRVideoConditionByReferenceLatent,
    'LTXDFRValidateLatentIndex_DFRSpatial': LTXDFRValidateLatentIndex,
    'LTXDFRValidateKeyframeIndex_DFRSpatial': LTXDFRValidateKeyframeIndex,
    'LTXDFRValidateGeneratedKeyframeSlots_DFRSpatial': LTXDFRValidateGeneratedKeyframeSlots,
    'LTXDFRValidateReferenceLatent_DFRSpatial': LTXDFRValidateReferenceLatent,
    'LTXDFRExtractCleanLatent_DFRSpatial': LTXDFRExtractCleanLatent,
    'LTXDFRExtractGeneratedKeyframes_DFRSpatial': LTXDFRExtractGeneratedKeyframes,
    'LTXDFRPrepareAutoFullDecodeScheduleC26B_DFRSpatial': LTXDFRPrepareAutoFullDecodeScheduleC26B,
    'LTXDFRPrepareExactFullDecodeScheduleC26B_DFRSpatial': LTXDFRPrepareExactFullDecodeScheduleC26B,
    'LTXDFRExecuteExactFullDecodeC26B_DFRSpatial': LTXDFRExecuteExactFullDecodeC26B,
    'LTXDFRValidateExactFullDecodeC26B_DFRSpatial': LTXDFRValidateExactFullDecodeC26B,
    'LTXDFRPrepareExactStage5DualStreamPortC25_DFRSpatial': LTXDFRPrepareExactStage5DualStreamPortC25,
    'LTXDFRExecuteExactStage5SingleTileC25_DFRSpatial': LTXDFRExecuteExactStage5SingleTileC25,
    'LTXDFRValidateExactStage5SingleTileC25_DFRSpatial': LTXDFRValidateExactStage5SingleTileC25,
    'LTXDFRPrepareExactDeterministicDualStreamPortC24B_DFRSpatial': LTXDFRPrepareExactDeterministicDualStreamPortC24B,
    'LTXDFRExecuteExactDeterministicStagesC24B_DFRSpatial': LTXDFRExecuteExactDeterministicStagesC24B,
    'LTXDFRValidateExactDeterministicStagesC24B_DFRSpatial': LTXDFRValidateExactDeterministicStagesC24B,
    'LTXDFRPrepareDeterministicDualStreamBlockPortU34B3B2_DFRSpatial': LTXDFRPrepareDeterministicDualStreamBlockPortU34B3B2,
    'LTXDFRValidateDeterministicDualStreamBlockPortU34B3B2_DFRSpatial': LTXDFRValidateDeterministicDualStreamBlockPortU34B3B2,
    'LTXDFRPrepareDeterministicStageOfficialPreflightU34B3B1_DFRSpatial': LTXDFRPrepareDeterministicStageOfficialPreflightU34B3B1,
    'LTXDFRValidateDeterministicStageOfficialPreflightU34B3B1_DFRSpatial': LTXDFRValidateDeterministicStageOfficialPreflightU34B3B1,
    'LTXDFRPrepareDeterministicDecoderKeyframeContextU34B3A_DFRSpatial': LTXDFRPrepareDeterministicDecoderKeyframeContextU34B3A,
    'LTXDFRValidateDeterministicDecoderKeyframeContextU34B3A_DFRSpatial': LTXDFRValidateDeterministicDecoderKeyframeContextU34B3A,
    'LTXDFRPrepareJointDecoderAttentionU34B2_DFRSpatial': LTXDFRPrepareJointDecoderAttentionU34B2,
    'LTXDFRValidateJointDecoderAttentionU34B2_DFRSpatial': LTXDFRValidateJointDecoderAttentionU34B2,
    'LTXDFRExtractDecoderKeyframeCheckpointWeightsU34B1B_DFRSpatial': LTXDFRExtractDecoderKeyframeCheckpointWeightsU34B1B,
    'LTXDFRValidateDecoderKeyframeCheckpointWeightsU34B1B_DFRSpatial': LTXDFRValidateDecoderKeyframeCheckpointWeightsU34B1B,
    'LTXDFRPrepareDecoderKeyframeSubstrateU34B1_DFRSpatial': LTXDFRPrepareDecoderKeyframeSubstrateU34B1,
    'LTXDFRValidateDecoderKeyframeSubstrateU34B1_DFRSpatial': LTXDFRValidateDecoderKeyframeSubstrateU34B1,
    'LTXDFRProbeDecoderKeyframeRuntimeU34A_DFRSpatial': LTXDFRProbeDecoderKeyframeRuntimeU34A,
    'LTXDFRValidateFinalDecoderBridgeU33B_DFRSpatial': LTXDFRValidateFinalDecoderBridgeU33B,
    'LTXDFRDecodeFinalVideoWithContinuedRNGU33B_DFRSpatial': LTXDFRDecodeFinalVideoWithContinuedRNGU33B,
    'LTXDFRProbeFinalDecoderRuntimeU33A_DFRSpatial': LTXDFRProbeFinalDecoderRuntimeU33A,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDFRExperimentalBlockStreamingDecode_DFRSpatial": "LTX DFR: Experimental Block Streaming Video Decode",
    "LTXSpatialDFRStage1PreviewDecode_DFRSpatial": "LTX Spatial DFR: Stage 1 Preview Decode",
    "LTXDFRSpatialDFRVideoDecode_DFRSpatial": "LTX DFR: Spatial DFR Video Decode",
    "LTXDFRPrepareAutoFullDecodeScheduleC26B_DFRSpatial": "LTX DFR: Prepare Auto Full Decode Schedule (C26b)",
    "LTXDFRPrepareExactFullDecodeScheduleC26B_DFRSpatial": "LTX DFR: Prepare Exact Full Decode Schedule (C26b)",
    "LTXDFRExecuteExactFullDecodeC26B_DFRSpatial": "LTX DFR: Execute Exact Full Decode (C26b)",
    "LTXDFRValidateExactFullDecodeC26B_DFRSpatial": "LTX DFR TEST: Validate Exact Full Decode (C26b)",
    "LTXDFRPrepareExactStage5DualStreamPortC25_DFRSpatial": "LTX DFR: Prepare Exact Stage-5 Dual-Stream Port (C25)",
    "LTXDFRExecuteExactStage5SingleTileC25_DFRSpatial": "LTX DFR: Execute Exact Stage-5 Single Tile (C25)",
    "LTXDFRValidateExactStage5SingleTileC25_DFRSpatial": "LTX DFR TEST: Validate Exact Stage-5 Single Tile (C25)",
    "LTXDFRPrepareExactDeterministicDualStreamPortC24B_DFRSpatial": "LTX DFR: Prepare Exact Deterministic Dual-Stream Port (C24b)",
    "LTXDFRExecuteExactDeterministicStagesC24B_DFRSpatial": "LTX DFR: Execute Exact Deterministic Stages (C24b)",
    "LTXDFRValidateExactDeterministicStagesC24B_DFRSpatial": "LTX DFR TEST: Validate Exact Deterministic Stages (C24b)",
    "LTXDFRPrepareDeterministicDualStreamBlockPortU34B3B2_DFRSpatial": "[DEPRECATED C24] LTX DFR: Prepare Deterministic Dual-Stream Block Port",
    "LTXDFRValidateDeterministicDualStreamBlockPortU34B3B2_DFRSpatial": "[DEPRECATED C24] LTX DFR TEST: Validate Deterministic Dual-Stream Block Port",
    "LTXDFRPrepareDeterministicStageOfficialPreflightU34B3B1_DFRSpatial": "LTX DFR: Prepare Deterministic Stage Official Preflight (U3.4b3b1 C23b)",
    "LTXDFRValidateDeterministicStageOfficialPreflightU34B3B1_DFRSpatial": "LTX DFR TEST: Validate Deterministic Stage Official Preflight (U3.4b3b1 C23b)",
    "LTXDFRPrepareDeterministicDecoderKeyframeContextU34B3A_DFRSpatial": "LTX DFR: Prepare Deterministic Decoder Keyframe Context (U3.4b3a FIX)",
    "LTXDFRValidateDeterministicDecoderKeyframeContextU34B3A_DFRSpatial": "LTX DFR TEST: Validate Deterministic Decoder Keyframe Context (U3.4b3a FIX)",
    "LTXDFRPrepareJointDecoderAttentionU34B2_DFRSpatial": "LTX DFR: Prepare Joint Decoder Attention (U3.4b2)",
    "LTXDFRValidateJointDecoderAttentionU34B2_DFRSpatial": "LTX DFR TEST: Validate Joint Decoder Attention (U3.4b2)",
    "LTXDFRExtractDecoderKeyframeCheckpointWeightsU34B1B_DFRSpatial": "LTX DFR: Extract Decoder Keyframe Checkpoint Weights (U3.4b1b)",
    "LTXDFRValidateDecoderKeyframeCheckpointWeightsU34B1B_DFRSpatial": "LTX DFR TEST: Validate Decoder Keyframe Checkpoint Weights (U3.4b1b)",
    "LTXDFRPrepareDecoderKeyframeSubstrateU34B1_DFRSpatial": "LTX DFR: Prepare Decoder Keyframe Substrate (U3.4b1)",
    "LTXDFRValidateDecoderKeyframeSubstrateU34B1_DFRSpatial": "LTX DFR TEST: Validate Decoder Keyframe Substrate (U3.4b1)",
    "LTXDFRProbeDecoderKeyframeRuntimeU34A_DFRSpatial": "LTX DFR TEST: Probe Decoder Keyframe Runtime (U3.4a)",
    "LTXDFRValidateFinalDecoderBridgeU33B_DFRSpatial": "LTX DFR TEST: Validate Final Decoder Bridge (U3.3b)",
    "LTXDFRDecodeFinalVideoWithContinuedRNGU33B_DFRSpatial": "LTX DFR: Decode Final Video With Continued RNG (U3.3b)",
    "LTXDFRProbeFinalDecoderRuntimeU33A_DFRSpatial": "LTX DFR TEST: Probe Final Decoder Runtime (U3.3a)",
    "LTXDFRPrepareOfficialFinalDecode_DFRSpatial": "LTX DFR: Prepare Official Final Decode",
    "LTXDFRPrepareStage2DecoderHandoff_DFRSpatial": "LTX DFR TEST: Prepare Final Decoder Handoff (U3.1 Atomic)",
    "LTXDFRBuildFinalDecodeKeyframes_DFRSpatial": "LTX DFR TEST: Build Final Decode Keyframes (U3.2 Atomic)",
    "LTXDFRValidateFinalDecodeKeyframes_DFRSpatial": "LTX DFR TEST: Validate Final Decode Keyframes (U3.2)",
    "LTXDFRValidateStage2DecoderHandoff_DFRSpatial": "LTX DFR TEST: Validate Final Decoder Handoff (U3.1)",
    "LTXDFRConditioningResizeCenterCrop_DFRSpatial": "LTX DFR: Conditioning Resize + Center Crop",
    "LTXDFRResolveDFRCanvas_DFRSpatial": "LTX DFR: Resolve DFR Canvas",
    "LTXExperimentalResolveDenseSlotCanvas16_DFRSpatial": "LTX Experimental: Resolve Dense 16-Frame Slot Canvas",
    "LTXDFRValidateDFRCanvas_DFRSpatial": "LTX DFR TEST: Validate DFR Canvas",
    "LTXDFRSigmaSchedules_DFRSpatial": "LTX DFR: DFR Sigma Schedules",
    "LTXDFRValidateDFRSigmaSchedules_DFRSpatial": "LTX DFR TEST: Validate DFR Sigma Schedules",
    "LTXDFRGaussianNoiser_DFRSpatial": "LTX DFR: Gaussian Noiser (DFR)",
    "LTXDFRValidateGaussianNoiser_DFRSpatial": "LTX DFR TEST: Validate Gaussian Noiser",
    "LTXDFRCreateStage1AudioState_DFRSpatial": "LTX DFR: Create Stage 1 Audio State",
    "LTXDFRValidateStage1AudioState_DFRSpatial": "LTX DFR TEST: Validate Stage 1 Audio State",
    "LTXDFRAudioGaussianNoiser_DFRSpatial": "LTX DFR: Gaussian Noiser (Audio DFR)",
    "LTXDFRValidateAudioGaussianNoiser_DFRSpatial": "LTX DFR TEST: Validate Audio Gaussian Noiser",
    "LTXDFRStage1AVGaussianNoiser_DFRSpatial": "LTX DFR: Stage 1 AV Gaussian Noiser",
    "LTXDFRValidateStage1AVGaussianNoiser_DFRSpatial": "LTX DFR TEST: Validate Stage 1 AV Gaussian Noiser",
    "LTXDFRMaterializeStage1AVModelInput_DFRSpatial": "LTX DFR: Materialize Stage 1 AV Model Input",
    "LTXDFRValidateStage1AVModelInput_DFRSpatial": "LTX DFR TEST: Validate Stage 1 AV Model Input",
    "LTXDFRExecuteStage1AVDenoise_DFRSpatial": "LTX DFR: Execute Stage 1 AV Denoise",
    "LTXDFRValidateStage1AVDenoise_DFRSpatial": "LTX DFR TEST: Validate Stage 1 AV Denoise",
    "LTXDFRStage1AVEulerStepFromDenoised_DFRSpatial": "LTX DFR: Stage 1 AV Euler Step From Denoised",
    "LTXDFRValidateStage1AVEulerStep_DFRSpatial": "LTX DFR TEST: Validate Stage 1 AV Euler Step",
    "LTXDFRValidateDFRModelCapability_DFRSpatial": "LTX DFR: DFR Model Capability Preflight",
    "LTXDFRRunStage1AVLoop_DFRSpatial": "LTX DFR: Run Stage 1 AV Loop",
    "LTXDFRRunStage1SpatialDFR_DFRSpatial": "LTX DFR: Run Stage 1 Spatial DFR",
    "LTXDFRValidateStage1AVLoop_DFRSpatial": "LTX DFR TEST: Validate Stage 1 AV Loop",
    "LTXDFRPrepareStage2Handoff_DFRSpatial": "LTX DFR: Prepare Stage 2 Handoff",
    "LTXDFRValidateStage2Handoff_DFRSpatial": "LTX DFR TEST: Validate Stage 2 Handoff",
    "LTXDFRRunStage2SpatialUpscale_DFRSpatial": "LTX DFR: Run Stage 2 Spatial Upscale",
    "LTXDFRRunModularStage2SpatialUpscale_DFRSpatial": "LTX DFR: Run Modular Stage 2 Spatial Upscaler",
    "LTXDFRRunStage2SpatialUpscaleFromStage2HandoffTest_DFRSpatial": "LTX DFR: Run Stage 2 Spatial Upscaler From Stage 2 Handoff (TEST ONLY)",
    "LTXDFRValidateStage2SpatialUpscale_DFRSpatial": "LTX DFR TEST: Validate Stage 2 Spatial Upscale",
    "LTXDFRResolveStage2DetailingMetadata_DFRSpatial": "LTX DFR TEST: Resolve Stage 2 Detailing Metadata (Atomic)",
    "LTXDFRPrepareStage2DetailingModel_DFRSpatial": "LTX DFR: Prepare Stage 2 Detailing Model",
    "LTXDFRFinalizeStage2DFRConditioning_DFRSpatial": "LTX DFR: Finalize Stage 2 DFR Conditioning",
    "LTXDFRValidateStage2DFRConditioning_DFRSpatial": "LTX DFR TEST: Validate Stage 2 DFR Conditioning",
    "LTXDFRApplyStage2DetailingICLoRA_DFRSpatial": "LTX DFR TEST: Apply Stage 2 Detailing IC-LoRA (Atomic)",
    "LTXDFRValidateStage2DetailingICLoRA_DFRSpatial": "LTX DFR TEST: Validate Stage 2 Detailing IC-LoRA",
    "LTXDFRStage2AVGaussianNoiser_DFRSpatial": "LTX DFR: Stage 2 AV Gaussian Noiser",
    "LTXDFRValidateStage2AVGaussianNoiser_DFRSpatial": "LTX DFR TEST: Validate Stage 2 AV Gaussian Noiser",
    "LTXDFRExecuteStage2AVDenoise_DFRSpatial": "LTX DFR: Execute Stage 2 AV Denoise",
    "LTXDFRValidateStage2AVDenoise_DFRSpatial": "LTX DFR TEST: Validate Stage 2 AV Denoise",
    "LTXDFRStage2AVEulerStepFromDenoised_DFRSpatial": "LTX DFR: Stage 2 AV Euler Step From Denoised",
    "LTXDFRValidateStage2AVEulerStep_DFRSpatial": "LTX DFR TEST: Validate Stage 2 AV Euler Step",
    "LTXDFRRunStage2AVLoop_DFRSpatial": "LTX DFR: Run Stage 2 AV Loop",
    "LTXExperimentalRunStage2AVLoopExtraStep_DFRSpatial": "LTX Experimental: Run Stage 2 AV Loop + One Step",
    "LTXDFRValidateStage2AVLoop_DFRSpatial": "LTX DFR TEST: Validate Stage 2 AV Loop",
    "LTXDFRFinalizeStage2Output_DFRSpatial": "LTX DFR: Finalize Stage 2 Output / Trim",
    "LTXDFRSplitStage1AudioVideo_DFRSpatial": "LTX DFR: Split Stage 1 Audio / Video + Trim",
    "LTXDFRValidateStage2FinalOutput_DFRSpatial": "LTX DFR TEST: Validate Stage 2 Final Output",
    "LTXDFRTrimStage1DecodedAudio_DFRSpatial": "LTX DFR: Trim Stage 1 Decoded Audio",
    "LTXDFRTrimStage2DecodedAudio_DFRSpatial": "LTX DFR: Trim Stage 2 Decoded Audio",
    "LTXDFRValidateStage2DecodedAudioTrim_DFRSpatial": "LTX DFR TEST: Validate Final Decoded Audio Trim",
    "LTXDFRMaterializeComfyModelInput_DFRSpatial": "LTX DFR: Materialize Comfy Model Input",
    "LTXDFRValidateComfyModelInput_DFRSpatial": "LTX DFR TEST: Validate Comfy Model Input",
    "LTXDFRExecuteComfyDenoise_DFRSpatial": "LTX DFR: Execute Comfy Denoise",
    "LTXDFRPostProcessDenoised_DFRSpatial": "LTX DFR: Post Process Denoised",
    "LTXDFREulerStepFromDenoised_DFRSpatial": "LTX DFR: Euler Step From Denoised",
    "LTXDFRValidateEulerStep_DFRSpatial": "LTX DFR TEST: Validate Euler Step",
    "LTXDFRRunStage1DFRLoop_DFRSpatial": "LTX DFR: Run Stage 1 DFR Loop",
    "LTXDFRValidateStage1DFRLoop_DFRSpatial": "LTX DFR TEST: Validate Stage 1 DFR Loop",
    "LTXDFRVideoConditionByLatentIndex_DFRSpatial": "LTX DFR: VideoConditionByLatentIndex",
    "LTXDFRVideoConditionByKeyframeIndex_DFRSpatial": "LTX DFR: VideoConditionByKeyframeIndex",
    "LTXDFRVideoGeneratedKeyframeSlots_DFRSpatial": "LTX DFR: VideoGeneratedKeyframeSlots",
    "LTXDFRVideoConditionByReferenceLatent_DFRSpatial": "LTX DFR: VideoConditionByReferenceLatent",
    "LTXDFRValidateLatentIndex_DFRSpatial": "LTX DFR TEST: Validate Latent Index",
    "LTXDFRValidateKeyframeIndex_DFRSpatial": "LTX DFR TEST: Validate Keyframe Index",
    "LTXDFRValidateGeneratedKeyframeSlots_DFRSpatial": "LTX DFR TEST: Validate Generated Keyframe Slots",
    "LTXDFRValidateReferenceLatent_DFRSpatial": "LTX DFR TEST: Validate Reference Latent",
    "LTXDFRExtractCleanLatent_DFRSpatial": "LTX DFR TEST: Extract Clean Latent",
    "LTXDFRExtractGeneratedKeyframes_DFRSpatial": "LTX DFR TEST: Extract Generated Keyframes",
}


# Keep the stable node identifiers above intact for workflow compatibility, but
# present the completed workflow surface separately from the atomic development
# nodes that were useful while building and validating parity.
_PUBLIC_NODE_CATEGORIES = {
    "LTXDFRExperimentalBlockStreamingDecode_DFRSpatial": "LTX/DFR Spatial/Experimental",
    # Setup and user conditioning.
    "LTXDFRResolveDFRCanvas_DFRSpatial": "LTX/DFR Spatial/01 - Setup & Conditioning",
    "LTXDFRConditioningResizeCenterCrop_DFRSpatial": "LTX/DFR Spatial/01 - Setup & Conditioning",
    "LTXDFRValidateDFRModelCapability_DFRSpatial": "LTX/DFR Spatial/01 - Setup & Conditioning",
    "LTXDFRSigmaSchedules_DFRSpatial": "LTX/DFR Spatial/01 - Setup & Conditioning",
    "LTXDFRCreateStage1AudioState_DFRSpatial": "LTX/DFR Spatial/01 - Setup & Conditioning",
    "LTXDFRVideoConditionByLatentIndex_DFRSpatial": "LTX/DFR Spatial/01 - Setup & Conditioning",
    "LTXDFRVideoConditionByKeyframeIndex_DFRSpatial": "LTX/DFR Spatial/01 - Setup & Conditioning",
    "LTXDFRVideoGeneratedKeyframeSlots_DFRSpatial": "LTX/DFR Spatial/01 - Setup & Conditioning",
    # Complete Stage-1 operations and preview/output helpers.
    "LTXDFRRunStage1SpatialDFR_DFRSpatial": "LTX/DFR Spatial/02 - Stage 1",
    "LTXDFRSplitStage1AudioVideo_DFRSpatial": "LTX/DFR Spatial/02 - Stage 1",
    "LTXSpatialDFRStage1PreviewDecode_DFRSpatial": "LTX/DFR Spatial/02 - Stage 1",
    "LTXDFRTrimStage1DecodedAudio_DFRSpatial": "LTX/DFR Spatial/02 - Stage 1",
    # Complete Stage-2 operations.
    "LTXDFRRunStage2SpatialUpscale_DFRSpatial": "LTX/DFR Spatial/03 - Stage 2",
    "LTXDFRRunModularStage2SpatialUpscale_DFRSpatial": "LTX/DFR Spatial/03 - Stage 2",
    "LTXDFRRunStage2SpatialUpscaleFromStage2HandoffTest_DFRSpatial": "LTX/DFR Spatial/03 - Stage 2",
    "LTXDFRPrepareStage2DetailingModel_DFRSpatial": "LTX/DFR Spatial/03 - Stage 2",
    "LTXDFRFinalizeStage2DFRConditioning_DFRSpatial": "LTX/DFR Spatial/03 - Stage 2",
    "LTXDFRRunStage2AVLoop_DFRSpatial": "LTX/DFR Spatial/03 - Stage 2",
    "LTXDFRFinalizeStage2Output_DFRSpatial": "LTX/DFR Spatial/03 - Stage 2",
    # Final decode/output operations.
    "LTXDFRSpatialDFRVideoDecode_DFRSpatial": "LTX/DFR Spatial/04 - Decode & Output",
    "LTXDFRTrimStage2DecodedAudio_DFRSpatial": "LTX/DFR Spatial/04 - Decode & Output",
}


def _apply_node_menu_categories():
    development_root = "LTX/DFR Development/Spatial"
    old_root = "LTX/DFR Spatial/"

    for node_id, node_class in NODE_CLASS_MAPPINGS.items():
        public_category = _PUBLIC_NODE_CATEGORIES.get(node_id)
        if public_category is not None:
            node_class.CATEGORY = public_category
            continue

        old_category = str(getattr(node_class, "CATEGORY", "Atomic"))
        old_group = (
            old_category[len(old_root) :]
            if old_category.startswith(old_root)
            else old_category
        )
        node_class.CATEGORY = f"{development_root}/{old_group}"

        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id)
        if node_id.startswith("LTXExperimental"):
            marker = "[DEV/EXPERIMENTAL]"
        elif "TEST" in display_name.upper() or "Validate" in node_id or "Probe" in node_id:
            marker = "[DEV/TEST]"
        else:
            marker = "[DEV/ATOMIC]"
        if not display_name.startswith("[DEV/"):
            NODE_DISPLAY_NAME_MAPPINGS[node_id] = f"{marker} {display_name}"


_apply_node_menu_categories()

# Explicit non-parity workflow surface; keep it outside the atomic dev nodes.
from .same_resolution_refinement import LTXExtractStage1VideoLatent, LTXRunSameResolutionRefinement

NODE_CLASS_MAPPINGS.update({
    "LTXExtractStage1VideoLatent_DFRSpatial": LTXExtractStage1VideoLatent,
    "LTXRunSameResolutionRefinement_DFRSpatial": LTXRunSameResolutionRefinement,
})
NODE_DISPLAY_NAME_MAPPINGS.update({
    "LTXExtractStage1VideoLatent_DFRSpatial": "LTX DFR: Extract Stage 1 Video Latent",
    "LTXRunSameResolutionRefinement_DFRSpatial": "LTX DFR: Run Same-Resolution Refinement (Experimental)",
})
