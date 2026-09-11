"""Temporal-only Stage-2 model tiling; one shared AV trajectory and RNG."""
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
import logging

_active_plan = ContextVar('ltx_stage2_temporal_plan', default=None)
log = logging.getLogger(__name__)


def _helpers():
    if __package__ == 'DFR_Spatial':
        from DFR_Temporal import post_temporal_spatial_execution as execution
        from DFR_Temporal.post_temporal_spatial_conditioning import DFRPostTemporalSpatialModelTile
        from DFR_Temporal.temporal_tiles import _split_canvas_at_seams
    else:
        from ..DFR_Temporal import post_temporal_spatial_execution as execution
        from ..DFR_Temporal.post_temporal_spatial_conditioning import DFRPostTemporalSpatialModelTile
        from ..DFR_Temporal.temporal_tiles import _split_canvas_at_seams
    return execution, DFRPostTemporalSpatialModelTile, _split_canvas_at_seams


def make_temporal_plan(shape, seams, count, scale=8):
    from .decoder_official_tiling import DimensionInterval
    _, Tile, split = _helpers()
    frames, height, width = map(int, shape)
    seams = tuple(map(int, seams))
    if not seams or any(p <= 0 or p % scale for p in seams) or any(b <= a for a,b in zip(seams,seams[1:])):
        raise ValueError('Temporal tiling requires ordered positive keyframe seams on the latent grid')
    if seams[-1] != (frames-1)*scale:
        raise ValueError('Temporal tiling seams must end at the padded video endpoint')
    latent_seams = (0, *(p//scale for p in seams))
    overlap = latent_seams[1]+1
    intervals = split(latent_seams, count, overlap, frames)
    tiles = tuple(Tile(i, iv, DimensionInterval(0,height,0,0), DimensionInterval(0,width,0,0), True)
                  for i,iv in enumerate(intervals))
    return SimpleNamespace(latent_shape=(frames,height,width), tiles=tiles,
                           temporal_intervals=tuple(intervals), pixel_seams=seams)


@contextmanager
def stage2_temporal_tiling(model, bundle, video_state, temporal_tiles=1):
    from .av_execution import _model_looks_like_comfy_patcher
    from .latent_state import clone_or_create_official_state, ensure_token_state
    requested = int(temporal_tiles)
    if requested != temporal_tiles or not 0 <= requested <= 64:
        raise ValueError('temporal_tiles must be an integer between 0 and 64')
    source = bundle.stage_1_handoff
    count = int(getattr(source, 'temporal_tiles', 1)) if requested == 0 else requested
    prepared = None
    if count > 1:
        if not _model_looks_like_comfy_patcher(model):
            raise ValueError('Stage-2 temporal tiling currently requires the native Comfy MODEL backend')
        state = clone_or_create_official_state(video_state)
        tokens = ensure_token_state(state, target_samples=video_state['samples'], fps=float(source.fps))
        seams = tuple(getattr(source, 'temporal_seams', ()))
        if not seams:
            seams = tuple((tokens.get('generated_keyframe_layout') or {}).get('pixel_frame_indices', ()))
        scale = int(state.get('scale_factors', (8,32,32))[0])
        plan = make_temporal_plan(bundle.upscaled_video_latent.shape[2:], seams, count, scale)
        prepared = SimpleNamespace(tile_plan=plan, conditioning_fps=float(source.fps))
        del state, tokens  # Do not retain an extra full conditioning state across sampling.
        log.info('[LTX DFR Stage 2 temporal tiling] windows=%d; spatial=full; latent intervals=%s',
                 len(plan.tiles), [(iv.start,iv.end,iv.left_ramp) for iv in plan.temporal_intervals])
    token = _active_plan.set(prepared)
    try:
        yield prepared
    finally:
        _active_plan.reset(token)


def install_temporal_model_wrapper(options, reference_input, comfy_utils, video_state):
    prepared = _active_plan.get()
    if prepared is None:
        return
    execution, _, _ = _helpers()
    options['model_function_wrapper'] = execution._ComfyPackedSpatialTilingWrapper(
        prepared, reference_input, comfy_utils, options.get('model_function_wrapper'),
        execution._absolute_base_positions(video_state, fps=prepared.conditioning_fps))
