"""Adapt completed DFR stages to the existing, full-frame Stage-2 pipeline."""
from dataclasses import replace
import torch
from .dfr_layout import validate_layout, make_continuation_layout
from .stage2_handoff import DFRStage2Handoff, HANDOFF_VERSION
from .stage2_result import DFRStage2ResultHandoff, require_stage2_result_handoff
from .stage2_spatial import prepare_stage2_spatial_upscale_from_handoff


def normalize_spatial_source(source, dfr_layout=None):
    if isinstance(source, DFRStage2Handoff):
        if source.version != HANDOFF_VERSION:
            raise ValueError('Unsupported Stage-1 handoff version')
        boundary = source
    elif isinstance(source, DFRStage2ResultHandoff):
        source = require_stage2_result_handoff(source)
        decoder = source.decoder_handoff
        boundary = DFRStage2Handoff(
            HANDOFF_VERSION, decoder.seed, decoder.rng_state_after_stage2_av,
            decoder.rng_device_type, source.padded_video_latent,
            decoder.stage_2_generated_keyframes, source.stage_1_audio_latent,
            (), (), source.fps, source.duration_seconds,
            tuple(getattr(source, "temporal_seams", ())), int(getattr(source, "temporal_tiles", 1)))
    else:
        # Import lazily: the temporal package itself uses spatial helpers.
        if __package__ == 'DFR_Spatial':
            from DFR_Temporal.temporal_handoff import require_temporal_handoff
        else:
            from ..DFR_Temporal.temporal_handoff import require_temporal_handoff
        source = require_temporal_handoff(source)
        if source.completed_rounds < 1:
            raise ValueError('Connect a completed temporal round, not its input handoff')
        boundary = DFRStage2Handoff(
            HANDOFF_VERSION, source.seed, source.rng_state_before_temporal,
            source.rng_device_type, source.video_latent, source.carry_keyframes,
            source.stage_1_audio_latent, (), (), source.fps, source.duration_seconds,
            tuple(source.last_window_seams), 2 ** int(source.completed_rounds))
        # All carry planes, including padding-region anchors, continue through
        # sampling. Only the final output node trims to requested_frames.
        dfr_layout = make_continuation_layout(source.requested_frames, source.padded_frames,
                                             source.carry_positions, source.temporal_scale)
    embedded = getattr(source, "dfr_layout", None)
    if embedded is not None:
        dfr_layout = embedded
    if dfr_layout is None:
        raise ValueError('Stage-1 and Stage-2 sources require their matching dfr_layout input')
    layout = dict(validate_layout(dfr_layout))
    video, keys, audio = boundary.reserved_half_res_video, boundary.stage_1_generated_keyframes, boundary.stage_1_audio_latent
    if not torch.is_tensor(video) or video.ndim != 5 or not torch.is_tensor(keys) or keys.ndim != 5:
        raise ValueError('Source video and generated keyframes must be [B,C,T,H,W] tensors')
    if video.shape[:2] != keys.shape[:2] or video.shape[-2:] != keys.shape[-2:]:
        raise ValueError('Source video/keyframe batch, channel and spatial dimensions must match')
    if video.shape[0] != 1:
        raise ValueError('Modular Stage 2 supports batch size one')
    if video.shape[2] != layout['padded_latent_frames'] or keys.shape[2] != len(layout['pixel_frame_indices']):
        raise ValueError('Source video/keyframe count does not match dfr_layout; use the layout from the previous stage')
    if not torch.is_tensor(audio) or audio.ndim != 4:
        raise ValueError('Source audio must be [B,C,T,F]')
    if not torch.is_tensor(boundary.rng_state_after_stage1_av):
        raise ValueError('Source has no RNG continuation state')
    boundary = replace(boundary, dfr_layout=dict(layout), rng_state_after_stage1_av=boundary.rng_state_after_stage1_av.detach().cpu().clone())
    return boundary, layout


def run_modular_spatial_upscale(source_handoff, upscale_model, vae, dfr_layout=None):
    boundary, layout = normalize_spatial_source(source_handoff, dfr_layout)
    bundle, latent = prepare_stage2_spatial_upscale_from_handoff(boundary, upscale_model, vae)
    return bundle, latent, layout, float(boundary.fps)
