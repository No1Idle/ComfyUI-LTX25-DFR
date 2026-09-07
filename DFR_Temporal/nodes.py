"""ComfyUI nodes for Temporal-DFR entry, diagnostics, and round execution."""

from __future__ import annotations

from .temporal_handoff import (
    prepare_temporal_handoff_from_stage1,
    prepare_temporal_handoff_from_stage2,
)
from .temporal_upscale import prepare_temporal_latent_upscale
from .temporal_tiles import prepare_temporal_tile_plan
from .temporal_execution import run_temporal_dfr_round
from .temporal_output import (
    decode_temporal_dfr_video,
    finalize_temporal_output,
    trim_temporal_decoded_audio,
)
from .post_temporal_spatial_execution import run_post_temporal_spatial_dfr


class LTXDFRPrepareTemporalHandoffFromStage1:
    """Stage-1 -> Temporal-DFR entry boundary."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_1_handoff": ("LTX_DFR_STAGE2_HANDOFF",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
            },
            "optional": {
                "user_conditioning_state": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Optional Stage-1 official video state containing the encoded user image/keyframe "
                            "conditions. Connect it to preserve and rebase those guides in temporal tiles."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("LTX_DFR_TEMPORAL_HANDOFF", "STRING")
    RETURN_NAMES = ("temporal_handoff", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Temporal/T1 - Handoff"
    DESCRIPTION = (
        "Normalizes Stage-1 base video, generated slots, Stage-1 audio, slot positions and continued RNG state "
        "into the same Temporal-DFR handoff used by the official Stage-2 path. Performs no upsampling."
    )

    def prepare(self, stage_1_handoff, dfr_layout, user_conditioning_state=None):
        return prepare_temporal_handoff_from_stage1(
            stage_1_handoff,
            dfr_layout,
            user_conditioning_state=user_conditioning_state,
        )


class LTXDFRPrepareTemporalHandoffFromStage2:
    """Official Spatial Stage-2 -> Temporal-DFR entry boundary."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_2_handoff": ("LTX_DFR_STAGE2_RESULT_HANDOFF",),
                "dfr_layout": ("LTX_DFR_LAYOUT",),
            },
            "optional": {
                "user_conditioning_state": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Optional Stage-2 official conditioning state containing the encoded user image/keyframe "
                            "conditions. Connect it to preserve and rebase those guides in temporal tiles."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("LTX_DFR_TEMPORAL_HANDOFF", "STRING")
    RETURN_NAMES = ("temporal_handoff", "report")
    FUNCTION = "prepare"
    CATEGORY = "LTX/DFR Temporal/T1 - Handoff"
    DESCRIPTION = (
        "Official Temporal-DFR entry point after Spatial Stage 2. Preserves the padded Stage-2 video, "
        "Stage-2 generated DFR keyframes, their layout positions, frozen Stage-1 audio and the continued "
        "Gaussian-noiser RNG state. Performs no temporal upsampling."
    )

    def prepare(self, stage_2_handoff, dfr_layout, user_conditioning_state=None):
        return prepare_temporal_handoff_from_stage2(
            stage_2_handoff,
            dfr_layout,
            user_conditioning_state=user_conditioning_state,
        )


class LTXDFRRunTemporalLatentUpscale:
    """Oracle first operation of the next Temporal-DFR round: video latent x2 in time."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temporal_handoff": ("LTX_DFR_TEMPORAL_HANDOFF",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_TEMPORAL_UPSCALED_HANDOFF", "LATENT", "STRING")
    RETURN_NAMES = ("upscaled_temporal_handoff", "upscaled_video_latent", "report")
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Temporal/T2 - Latent Upscale"
    DESCRIPTION = (
        "Runs the native LTXVLatentUpsampler on the base VIDEO latent only, matching official Temporal DFR. "
        "Expected geometry is T -> 2T-1 with H/W unchanged. Carry keyframe tensors are NOT upscaled; their "
        "pixel-frame positions double and playback fps doubles. Transformer conditioning fps follows the official "
        "high-fps policy. Works from either the Stage-1 handoff or "
        "the official Stage-2 handoff."
    )

    def run(self, temporal_handoff, upscale_model, vae):
        return prepare_temporal_latent_upscale(temporal_handoff, upscale_model, vae)



class LTXDFRBuildTemporalTilePlan:
    """Oracle temporal seam-window plan for the current x2 round."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temporal_upscaled_handoff": ("LTX_DFR_TEMPORAL_UPSCALED_HANDOFF",),
            }
        }

    RETURN_TYPES = ("LTX_DFR_TEMPORAL_TILE_PLAN", "STRING")
    RETURN_NAMES = ("temporal_tile_plan", "report")
    FUNCTION = "build"
    CATEGORY = "LTX/DFR Temporal/T3 - Tile Plan"
    DESCRIPTION = (
        "Builds the official Temporal DFR seam-window plan for the current round. "
        "Each later tile contains a lead-in latent prefix (left_ramp) that is denoised for context "
        "but dropped during stitching. Windows expose global and local anchor/slot positions for the next phase."
    )

    def build(self, temporal_upscaled_handoff):
        return prepare_temporal_tile_plan(temporal_upscaled_handoff)


class LTXRunTemporalDFRRoundEulerPreview:
    """Run one complete temporal x2 round with the official sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "temporal_handoff": ("LTX_DFR_TEMPORAL_HANDOFF",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "vae": ("VAE",),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Keep 1.0 for the current distilled Temporal-DFR path.",
                    },
                ),
                "anchor_strength": (
                    "FLOAT",
                    {
                        "default": 0.95,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Strength of existing carry keyframes inside this temporal round. "
                            "0.95 is official; lower values allow more anchor denoising."
                        ),
                    },
                ),
                "carry_refined_anchors": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Experimental. When enabled, carry the denoised anchor-token outputs instead of "
                            "the original anchors. Duplicate seam anchors use the earlier tile's result."
                        ),
                    },
                ),
            },
            "optional": {
                "negative": (
                    "CONDITIONING",
                    {"tooltip": "Optional Comfy CFG extension; unused at cfg_scale=1.0."},
                ),
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Optional advanced override. When disconnected, uses the official temporal "
                            "schedule 0.975, 0.909375, 0.725, 0.421875, 0 (four transitions)."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("LTX_DFR_TEMPORAL_HANDOFF",)
    RETURN_NAMES = ("temporal_handoff",)
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Temporal/Temporal Round"
    DESCRIPTION = (
        "Runs one complete Temporal-DFR x2 round: temporal latent upscale, official seam-window plan, "
        "tile-local user/anchor/slot conditioning, continued shared Gaussian RNG, frozen Stage-1 audio, "
        "four official eta=0.5 Euler-Ancestral transitions per tile, stitching, and carry-forward keyframes. "
        "Step-noise uses the official independent per-tile seed and VIDEO-then-AUDIO draw order. The output "
        "may be fed back into this node for the second x2 round. Defaults preserve official 0.95 anchors; "
        "optional refined-anchor carry is an experimental low-resolution workflow extension."
    )

    def run(
        self,
        model,
        positive,
        temporal_handoff,
        upscale_model,
        vae,
        cfg_scale=1.0,
        anchor_strength=0.95,
        carry_refined_anchors=False,
        negative=None,
        sigmas=None,
    ):
        return (
            run_temporal_dfr_round(
                model=model,
                positive=positive,
                temporal_handoff=temporal_handoff,
                upscale_model=upscale_model,
                vae=vae,
                negative=negative,
                sigmas=sigmas,
                cfg_scale=float(cfg_scale),
                anchor_strength=float(anchor_strength),
                carry_refined_anchors=bool(carry_refined_anchors),
            ),
        )


class LTXRunPostTemporalSpatialDFRX2:
    """Run the terminal latent-only Spatial x2 epilogue after Temporal DFR."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "The MODEL produced by 'Apply Stage 2 Detailing IC-LoRA'. "
                            "The same x2 Pixel-Spatial detailing adapter is required here."
                        )
                    },
                ),
                "positive": ("CONDITIONING",),
                "temporal_handoff": (
                    "LTX_DFR_TEMPORAL_HANDOFF",
                    {
                        "tooltip": (
                            "A handoff after one or two completed Temporal-DFR rounds. "
                            "The output becomes terminal and cannot enter another temporal round."
                        )
                    },
                ),
                "spatial_upscale_model": (
                    "LATENT_UPSCALE_MODEL",
                    {"tooltip": "The LTX x2 spatial latent upsampler model."},
                ),
                "vae": (
                    "VAE",
                    {"tooltip": "Passed to Comfy's native LTXVLatentUpsampler."},
                ),
                "cfg_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.01,
                        "tooltip": "Keep 1.0 for the distilled DFR path.",
                    },
                ),
                "keyframe_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Denoise strength for every propagated temporal carry keyframe. "
                            "1.0 is official; lower values are the agreed latent-only low-resolution experiment."
                        ),
                    },
                ),
            },
            "optional": {
                "negative": (
                    "CONDITIONING",
                    {"tooltip": "Optional Comfy CFG extension; unused at cfg_scale=1.0."},
                ),
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": (
                            "Optional advanced override. When disconnected, uses the official Spatial "
                            "Stage-2 schedule 0.909375, 0.725, 0.421875, 0."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("LTX_DFR_TEMPORAL_HANDOFF",)
    RETURN_NAMES = ("terminal_handoff",)
    FUNCTION = "run"
    CATEGORY = "LTX/DFR Temporal/Post-Temporal Spatial x2"
    DESCRIPTION = (
        "Runs the terminal post-temporal Spatial x2 epilogue as one consolidated operation: spatially "
        "upsamples the completed video and all carried/user keyframes, assembles user -> carry -> x2-reference "
        "conditioning, applies fresh seed+2000 Gaussian noise, and executes one full-canvas three-step Euler "
        "trajectory whose transformer prediction is tiled and blended at every step. Stage-1 audio remains "
        "frozen for cross-modal attention. This latent-only variant deliberately avoids the official RGB "
        "decode/Lanczos/re-encode carry path. Connect the output directly to Finalize Output / Split AV."
    )

    def run(
        self,
        model,
        positive,
        temporal_handoff,
        spatial_upscale_model,
        vae,
        cfg_scale=1.0,
        keyframe_strength=1.0,
        negative=None,
        sigmas=None,
    ):
        return (
            run_post_temporal_spatial_dfr(
                model=model,
                positive=positive,
                temporal_handoff=temporal_handoff,
                spatial_upscale_model=spatial_upscale_model,
                vae=vae,
                negative=negative,
                sigmas=sigmas,
                cfg_scale=float(cfg_scale),
                keyframe_strength=float(keyframe_strength),
            ),
        )


class LTXFinalizeTemporalDFROutput:
    """Trim final temporal video and expose the preserved Stage-1 audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temporal_handoff": (
                    "LTX_DFR_TEMPORAL_HANDOFF",
                    {"tooltip": "Output from the final Temporal DFR round."},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "FLOAT")
    RETURN_NAMES = ("final_video_latent", "final_audio_latent", "output_fps")
    FUNCTION = "finalize"
    CATEGORY = "LTX/DFR Temporal/Final Output"
    DESCRIPTION = (
        "Trims the final temporal video latent from the padded working canvas to the requested duration, "
        "returns the unchanged Stage-1 audio latent for LTXV Audio VAE Decode, and exposes the doubled "
        "playback FPS. The audio latent is cropped only after decoding."
    )

    def finalize(self, temporal_handoff):
        return finalize_temporal_output(temporal_handoff)


class LTXTemporalDFRVideoDecode:
    """Final keyframe-aware DiffVAE decode for a Temporal DFR handoff."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths

            vae_choices = folder_paths.get_filename_list("vae")
        except Exception:
            vae_choices = []
        return {
            "required": {
                "temporal_handoff": (
                    "LTX_DFR_TEMPORAL_HANDOFF",
                    {"tooltip": "Output from the final Temporal DFR round."},
                ),
                "final_video_latent": (
                    "LATENT",
                    {"tooltip": "Video output from 'Finalize Temporal Output / Split AV'."},
                ),
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
                        "tooltip": "Manual mode only. Temporal overlap is derived internally.",
                    },
                ),
                "tile_height": (
                    "INT",
                    {
                        "default": 416,
                        "min": 320,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "Manual mode only. Spatial overlap is derived internally.",
                    },
                ),
                "tile_width": (
                    "INT",
                    {
                        "default": 544,
                        "min": 320,
                        "max": 16384,
                        "step": 32,
                        "tooltip": "Manual mode only. Spatial overlap is derived internally.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "LTX/DFR Temporal/Final Output"
    DESCRIPTION = (
        "Drops carry keyframes beyond the requested temporal canvas, restores the shared post-temporal "
        "Gaussian RNG state, and runs the complete validated keyframe-aware DiffVAE decoder with official "
        "auto tiling or manual tile sizes. Works after either one x2 round or the second x2/x4 round."
    )

    def decode(
        self,
        temporal_handoff,
        final_video_latent,
        vae,
        vae_name,
        use_auto_tiling=True,
        tile_frames=104,
        tile_height=416,
        tile_width=544,
    ):
        return (
            decode_temporal_dfr_video(
                temporal_handoff,
                final_video_latent,
                vae,
                str(vae_name),
                use_auto_tiling=bool(use_auto_tiling),
                tile_frames=int(tile_frames),
                tile_height=int(tile_height),
                tile_width=int(tile_width),
            ),
        )


class LTXTrimTemporalDFRDecodedAudio:
    """Crop decoded Stage-1 audio to the final temporal video duration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temporal_handoff": (
                    "LTX_DFR_TEMPORAL_HANDOFF",
                    {"tooltip": "Output from the final Temporal DFR round."},
                ),
                "decoded_audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Decode final_audio_latent from 'Finalize Temporal Output / Split AV' with "
                            "LTXV Audio VAE Decode, then connect the result here."
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("final_audio",)
    FUNCTION = "trim"
    CATEGORY = "LTX/DFR Temporal/Final Output"
    DESCRIPTION = (
        "Crops the decoded Stage-1 audio to round((requested temporal frames / output FPS) * sample_rate), "
        "matching the official final audio epilogue after one x2 or two x2/x4 temporal rounds."
    )

    def trim(self, temporal_handoff, decoded_audio):
        return (trim_temporal_decoded_audio(temporal_handoff, decoded_audio),)

NODE_CLASS_MAPPINGS = {
    "LTXDFRPrepareTemporalHandoffFromStage1_DFRTemporal": LTXDFRPrepareTemporalHandoffFromStage1,
    "LTXDFRPrepareTemporalHandoffFromStage2_DFRTemporal": LTXDFRPrepareTemporalHandoffFromStage2,
    "LTXRunTemporalDFRRoundEulerPreview_DFRTemporal": LTXRunTemporalDFRRoundEulerPreview,
    "LTXRunPostTemporalSpatialDFRX2_DFRTemporal": LTXRunPostTemporalSpatialDFRX2,
    "LTXFinalizeTemporalDFROutput_DFRTemporal": LTXFinalizeTemporalDFROutput,
    "LTXTemporalDFRVideoDecode_DFRTemporal": LTXTemporalDFRVideoDecode,
    "LTXTrimTemporalDFRDecodedAudio_DFRTemporal": LTXTrimTemporalDFRDecodedAudio,
}
DEV_NODE_CLASS_MAPPINGS = {
    "LTXDFRRunTemporalLatentUpscale_DFRTemporal": LTXDFRRunTemporalLatentUpscale,
    "LTXDFRBuildTemporalTilePlan_DFRTemporal": LTXDFRBuildTemporalTilePlan,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDFRPrepareTemporalHandoffFromStage1_DFRTemporal": "LTX DFR Temporal: Prepare Handoff From Stage 1",
    "LTXDFRPrepareTemporalHandoffFromStage2_DFRTemporal": "LTX DFR Temporal: Prepare Handoff From Stage 2",
    "LTXDFRRunTemporalLatentUpscale_DFRTemporal": "LTX DFR Temporal: Run Temporal Latent Upscale x2",
    "LTXDFRBuildTemporalTilePlan_DFRTemporal": "LTX DFR Temporal: Build Temporal Tile Plan",
    "LTXRunTemporalDFRRoundEulerPreview_DFRTemporal": "LTX DFR Temporal: Run Temporal Round (Official Ancestral)",
    "LTXRunPostTemporalSpatialDFRX2_DFRTemporal": "LTX DFR Temporal: Run Post-Temporal Spatial x2",
    "LTXFinalizeTemporalDFROutput_DFRTemporal": "LTX DFR Temporal: Finalize Output / Split AV",
    "LTXTemporalDFRVideoDecode_DFRTemporal": "LTX DFR Temporal: Video Decode",
    "LTXTrimTemporalDFRDecodedAudio_DFRTemporal": "LTX DFR Temporal: Trim Decoded Audio",
}


# The whole-round nodes are the supported workflow surface. Keep the two
# independently runnable construction steps available for diagnostics, but move
# them out of the normal Temporal DFR menu.
_PUBLIC_NODE_CATEGORIES = {
    "LTXDFRPrepareTemporalHandoffFromStage1_DFRTemporal": "LTX/DFR Temporal/01 - Input Handoff",
    "LTXDFRPrepareTemporalHandoffFromStage2_DFRTemporal": "LTX/DFR Temporal/01 - Input Handoff",
    "LTXRunTemporalDFRRoundEulerPreview_DFRTemporal": "LTX/DFR Temporal/02 - Processing",
    "LTXRunPostTemporalSpatialDFRX2_DFRTemporal": "LTX/DFR Temporal/02 - Processing",
    "LTXFinalizeTemporalDFROutput_DFRTemporal": "LTX/DFR Temporal/03 - Decode & Output",
    "LTXTemporalDFRVideoDecode_DFRTemporal": "LTX/DFR Temporal/03 - Decode & Output",
    "LTXTrimTemporalDFRDecodedAudio_DFRTemporal": "LTX/DFR Temporal/03 - Decode & Output",
}


def _apply_node_menu_categories():
    for node_id, node_class in NODE_CLASS_MAPPINGS.items():
        public_category = _PUBLIC_NODE_CATEGORIES.get(node_id)
        if public_category is not None:
            node_class.CATEGORY = public_category
            continue

        node_class.CATEGORY = "LTX/DFR Development/Temporal/Atomic"
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(node_id, node_id)
        if not display_name.startswith("[DEV/"):
            NODE_DISPLAY_NAME_MAPPINGS[node_id] = f"[DEV/ATOMIC] {display_name}"


_apply_node_menu_categories()
