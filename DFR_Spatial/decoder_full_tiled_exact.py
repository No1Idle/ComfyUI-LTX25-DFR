"""C26b / U3.4b5: exact frozen tiled keyframe-aware final DiffVAE decode.

C26b replaces the rejected C26 renderer.  Tile geometry, masks, temporal-group
handoff, keyframe-plane selection, two time origins, ghost handling and RNG draw
order are direct ports of the frozen LTX-core sources.  The neural blocks remain
the already runtime-validated C24b deterministic dual stream and the official
C25 CHUNKED_EAGER deferred-Stage-4, four-way-width Stage-5 mapping.

Current checkpoint contract is intentionally narrow and already validated by C25:
``model_output_type='x0'`` with one decoder inference timestep.  Therefore the
frozen decoder leaves ``x_t_init=None`` at group scope and draws VIDEO pixel noise
per tile, followed by that tile's separate KEYFRAME pixel noise, from the one
continued post-Stage2 generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .decoder_bridge import restore_decoder_generator
from .decoder_det_dual_stream_exact import (
    DFRDeterministicStageExecution,
    _KeyframeStream,
    _load_comfy_vae_for_execution,
    _run_det_stage_blocks_with_keyframes_exact,
    _sample_parameter_checksum,
)
from .decoder_det_stage_context import DFRDeterministicDecoderKeyframeContext
from .decoder_det_stage_official_preflight import keyframe_clip_times_official
from .decoder_handoff import DFRStage2DecoderHandoff
from .decoder_stage5_dual_stream_exact import (
    DFRExactStage5DualStreamPort,
)
from .decoder_stage5_chunked_exact import (
    OFFICIAL_STAGE5_W_CHUNKS,
    build_deferred_stage5_inputs_exact,
    forward_stage5_chunked_step_exact,
    stage5_grid_shape_from_stage4,
)
from .decoder_official_tiling import (
    DimensionSizeConfig,
    Tile,
    TileSizeConfig,
    all_stages_min_tile_size,
    compute_tile_halos,
    compute_tile_min_size,
    crop_pixels_to_content,
    crop_trailing_context_natten_pad,
    group_tiles_by_temporal_slice,
    masks_are_complementary,
    pixel_tile_shape,
    prepare_tile_schedule,
    recommended_chunked_eager_keyframe_decode_tiling_config,
    recommended_pixel_overlaps,
    scale_by_masks_1d,
    slice_stage4_tile,
    stage4_thw_from_latent,
    stage4_to_pixel_scale_factors,
    _weight_floor,
)


C26B_SCHEDULE_VERSION = 2
C26B_EXECUTION_VERSION = 2


@dataclass(frozen=True)
class DFROfficialTiledDecodeSchedule:
    version: int
    source_contract: str
    tiling_config: TileSizeConfig
    content_latent_shape: tuple[int, ...]
    content_stage4_shape: tuple[int, int, int]
    resident_stage4_feature_shape: tuple[int, ...]
    full_pixel_shape_bcfhw: tuple[int, ...]
    stage_min_latent_size: tuple[int, int, int]
    tile_min_stage4_size: tuple[int, int, int]
    tile_halos: tuple[tuple[int, int, int], tuple[int, int, int]]
    pixel_scale_from_stage4: tuple[int, int, int]
    required_temporal_overlap: int
    required_spatial_overlap: int
    stage4_kernel: tuple[int, int, int]
    stage4_depth: int
    stage5_kernel: tuple[int, int, int]
    stage5_depth: int
    upsample3_stride: tuple[int, int, int]
    patch_size: int
    ghost_pad_latent_frames: int
    tiles: tuple[Tile, ...]
    group_slices: tuple[tuple[int, int], ...]
    tile_plane_indices: tuple[tuple[int, ...], ...]
    complementary_masks: bool
    total_tiles: int
    temporal_groups: int


@dataclass(frozen=True)
class DFROfficialTiledDecodeExecution:
    version: int
    source_contract: str
    execution_mode: str
    device: str
    dtype: str
    model_output_type: str
    inference_timesteps: tuple[float, ...]
    output_image_shape: tuple[int, ...]
    output_pixel_shape_bcfhw: tuple[int, ...]
    total_tiles: int
    rendered_tiles: int
    temporal_groups: int
    rendered_groups: int
    complementary_masks: bool
    stage5_block_count: int
    stage5_blocks_executed_total: int
    min_planes_per_tile: int
    max_planes_per_tile: int
    tile_plane_indices: tuple[tuple[int, ...], ...]
    stage4_time_origins: tuple[float, ...]
    pixel_time_origins: tuple[float, ...]
    generator_state_restored: bool
    generator_state_advanced: bool
    generator_initial_state: torch.Tensor
    generator_final_state: torch.Tensor
    rng_device_type: str
    cpu_rng_unchanged: bool
    cuda_rng_unchanged: bool
    decoder_parameters_unchanged: bool
    decoder_mutated: bool


def planes_for_tile_official(
    pixel_frame_indices: torch.Tensor,
    frame_lo: int,
    frame_hi: int,
    *,
    clip_start_frame: int = 0,
) -> torch.Tensor:
    """Direct frozen ``keyframes.planes_for_tile`` port."""
    frame_lo = frame_lo + clip_start_frame
    frame_hi = frame_hi + clip_start_frame
    indices = pixel_frame_indices.to(torch.int64)
    keep = (indices >= frame_lo) & (indices <= frame_hi)
    before = indices < frame_lo
    if bool(before.any()):
        keep[int(torch.where(before, indices, torch.full_like(indices, -1)).argmax())] = True
    after = indices > frame_hi
    if bool(after.any()):
        sentinel = int(indices.max()) + 1
        keep[int(torch.where(after, indices, torch.full_like(indices, sentinel)).argmin())] = True
    return keep


def _kernel3(module: Any) -> tuple[int, int, int]:
    k = tuple(int(v) for v in module.attn.kernel_size)
    if len(k) != 3:
        raise ValueError(f"Expected 3-D attention kernel, got {k}.")
    return k


def _runtime_decoder_geometry(decoder: Any):
    if len(decoder.det_stages) != 4:
        raise ValueError(f"C26b expects four deterministic stages, got {len(decoder.det_stages)}.")
    if not decoder.diff_blocks:
        raise ValueError("C26b requires non-empty Stage-5 diffusion blocks.")
    stage_kernels = tuple(_kernel3(stage[0]) for stage in decoder.det_stages)
    stage4_kernel = stage_kernels[3]
    stage5_kernel = _kernel3(decoder.diff_blocks[0])
    if any(_kernel3(block) != stage5_kernel for block in decoder.diff_blocks):
        raise ValueError("Loaded Stage-5 blocks do not share one kernel size; frozen C26b contract violated.")
    strides = tuple(tuple(int(v) for v in up.stride) for up in decoder.upsamples)
    if len(strides) != 4:
        raise ValueError(f"C26b expects four upsamplers, got {len(strides)}.")
    patch_size = int(decoder.patch_size)
    if patch_size < 1:
        raise ValueError(f"Invalid decoder patch_size={patch_size}.")
    return stage_kernels, stage5_kernel, strides, patch_size


def _module_state_bytes(module: Any) -> int:
    return sum(int(t.nbytes) for t in module.state_dict().values() if torch.is_tensor(t))


def _module_resident_bytes(module: Any, device: torch.device) -> int:
    total = 0
    for tensor in module.state_dict().values():
        if torch.is_tensor(tensor) and tensor.device == device:
            total += int(tensor.nbytes)
    return total


def _comfy_auto_tiling_free_bytes(
    vae: Any,
    decoder: Any,
    stage4_video: torch.Tensor,
    *,
    decoder_model_bytes: int,
) -> tuple[int, int, int, int, int]:
    """Translate current Comfy residency into the official pre-decode budget.

    The official pipeline recommends tiles before loading decoder weights and
    before creating the resident Stage-4 video feature.  This Comfy graph must
    recommend after C24b produced that feature, so those two owned allocations
    are credited back before the unchanged official budget formula subtracts
    them.  Other live tensors remain charged.
    """
    device = stage4_video.device
    if device.type != "cuda":
        raise ValueError(
            "Automatic DiffVAE tiling currently requires a CUDA Stage-4 execution. "
            "Use the unchanged manual schedule node on other devices."
        )
    try:
        import comfy.model_management as model_management
    except Exception as exc:  # pragma: no cover - only reachable outside ComfyUI
        raise RuntimeError("ComfyUI model-management runtime is unavailable for automatic tiling.") from exc

    current_free = int(model_management.get_free_memory(device))
    total_memory = int(model_management.get_total_memory(device))
    resident_decoder = min(
        int(decoder_model_bytes),
        _module_resident_bytes(decoder, device),
    )
    patcher = getattr(vae, "patcher", None)
    loaded_size = getattr(patcher, "loaded_size", None)
    if callable(loaded_size):
        resident_decoder = max(
            resident_decoder,
            min(int(decoder_model_bytes), max(0, int(loaded_size()))),
        )
    resident_stage4 = int(stage4_video.numel()) * int(stage4_video.element_size())
    planning_free = min(total_memory, current_free + resident_decoder + resident_stage4)
    return planning_free, current_free, total_memory, resident_decoder, resident_stage4


def _slice_pair(sl: slice, length: int) -> tuple[int, int]:
    start, stop, _ = sl.indices(length)
    return int(start), int(stop)


def _tile_short(tile: Tile, stage4_shape: tuple[int, int, int], pixel_shape: tuple[int, ...]) -> str:
    st = _slice_pair(tile.in_coords[1], stage4_shape[0])
    sh = _slice_pair(tile.in_coords[2], stage4_shape[1])
    sw = _slice_pair(tile.in_coords[3], stage4_shape[2])
    pt = _slice_pair(tile.out_coords[2], pixel_shape[2])
    ph = _slice_pair(tile.out_coords[3], pixel_shape[3])
    pw = _slice_pair(tile.out_coords[4], pixel_shape[4])
    return f"s4=T{st}/H{sh}/W{sw}->px=T{pt}/H{ph}/W{pw}"


def prepare_official_tiled_decode_schedule(
    vae: Any,
    deterministic_execution: DFRDeterministicStageExecution,
    deterministic_context: DFRDeterministicDecoderKeyframeContext,
    *,
    tile_frames: int = 104,
    temporal_overlap: int = 40,
    tile_height: int = 416,
    tile_width: int = 544,
    spatial_overlap: int = 160,
) -> tuple[DFROfficialTiledDecodeSchedule, str]:
    if not isinstance(deterministic_execution, DFRDeterministicStageExecution):
        raise ValueError(
            f"Expected DFRDeterministicStageExecution, got {type(deterministic_execution).__name__}."
        )
    if not isinstance(deterministic_context, DFRDeterministicDecoderKeyframeContext):
        raise ValueError(
            f"Expected DFRDeterministicDecoderKeyframeContext, got {type(deterministic_context).__name__}."
        )
    if deterministic_execution.stage5_called:
        raise ValueError("C26b schedule requires the validated C24b state that stops before Stage 5.")

    first_stage_model = getattr(vae, "first_stage_model", None)
    decoder = getattr(first_stage_model, "decoder", None)
    if decoder is None:
        raise ValueError("Connected VAE does not expose first_stage_model.decoder.")

    stage_kernels, stage5_kernel, strides, patch_size = _runtime_decoder_geometry(decoder)
    if strides != tuple(tuple(int(v) for v in s) for s in deterministic_context.upsample_strides):
        raise ValueError(f"Runtime upsample ladder {strides} != validated C22b ladder {deterministic_context.upsample_strides}.")
    if patch_size != int(deterministic_context.patch_size):
        raise ValueError(f"Runtime patch_size={patch_size} != validated context={deterministic_context.patch_size}.")

    stage4_depth = int(len(decoder.det_stages[3]))
    stage5_depth = int(len(decoder.diff_blocks))
    up3_stride = strides[3]
    tile_min = compute_tile_min_size(stage_kernels[3], stage5_kernel, up3_stride)
    tile_halos = compute_tile_halos(stage_kernels[3], stage4_depth, stage5_kernel, stage5_depth, up3_stride)
    pixel_scale = stage4_to_pixel_scale_factors(up3_stride, patch_size)
    req_t, req_hw = recommended_pixel_overlaps(tile_halos, pixel_scale)

    # C24b did not perform a latent size-floor step.  For the validated target
    # this is safe; if a future input needs padding, stop instead of guessing how
    # to splice a post-hoc pad into the already-computed C24b features.
    upsamples_for_floor = tuple((stride, 1) for stride in strides)
    stage_min = all_stages_min_tile_size(stage_kernels, upsamples_for_floor, stage5_kernel)
    b, _c, latent_t, latent_h, latent_w = (int(v) for v in deterministic_execution.input_video_shape)
    if b != 1:
        raise ValueError(
            f"C26b is locked to the official decode_video batch-1 contract used by this DFR path; got batch={b}."
        )
    if latent_t < stage_min[0] or latent_h < stage_min[1] or latent_w < stage_min[2]:
        raise ValueError(
            "The current final latent requires the official pre-stage size floor, but C24b was already executed "
            f"without it: latent={(latent_t, latent_h, latent_w)}, required={stage_min}. Stop here; do not infer a repair."
        )

    expected_ghost = (stage_kernels[0][0] // 2) * 2
    if int(deterministic_execution.ghost_pad_latent_frames) != int(expected_ghost):
        raise ValueError(
            f"C24b ghost pad={deterministic_execution.ghost_pad_latent_frames} != frozen decoder geometry {expected_ghost}."
        )

    content_s4 = stage4_thw_from_latent(strides, latent_t, latent_h, latent_w, drop_leading_frame=True)
    resident_s4 = tuple(int(v) for v in deterministic_execution.stage4_input_video.shape)
    if resident_s4[0] != b or resident_s4[2] != content_s4[1] or resident_s4[3] != content_s4[2]:
        raise ValueError(
            f"C24b resident Stage-4 feature geometry {resident_s4} is incompatible with content Stage-4 {content_s4}."
        )
    if resident_s4[1] < content_s4[0]:
        raise ValueError(f"Resident Stage-4 T={resident_s4[1]} is shorter than content T={content_s4[0]}.")

    time_scale = int(deterministic_context.temporal_scale_schedule[0])
    frames = (latent_t - 1) * time_scale + 1
    height, width = (int(v) for v in deterministic_context.final_pixel_spatial_shape)
    out_channels = int(decoder.out_channels)
    full_shape = (b, out_channels, frames, height, width)

    tiling_config = TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=int(tile_frames), overlap=int(temporal_overlap)),
        height=DimensionSizeConfig(tile_size=int(tile_height), overlap=int(spatial_overlap)),
        width=DimensionSizeConfig(tile_size=int(tile_width), overlap=int(spatial_overlap)),
    )
    tiles = prepare_tile_schedule(
        torch.Size([b, int(deterministic_execution.input_video_shape[1]), *content_s4]),
        tiling_config,
        upsample3_stride=up3_stride,
        patch_size=patch_size,
        min_tile_size=tile_min,
        tile_halos=tile_halos,
    )
    if not tiles:
        raise RuntimeError("Frozen prepare_tile_schedule returned no tiles.")
    complementary = masks_are_complementary(tiles, full_shape)
    groups = group_tiles_by_temporal_slice(list(tiles))
    group_slices = tuple(_slice_pair(group[0].out_coords[2], full_shape[2]) for group in groups)

    pixel_indices = torch.tensor(deterministic_context.keyframe_pixel_frame_indices, dtype=torch.long, device="cpu")
    plane_sets: list[tuple[int, ...]] = []
    for tile in tiles:
        pixel_lo, pixel_hi = _slice_pair(tile.out_coords[2], full_shape[2])
        keep = planes_for_tile_official(pixel_indices, pixel_lo, pixel_hi - 1, clip_start_frame=0)
        if not bool(keep.any()):
            raise RuntimeError("Frozen planes_for_tile selected no keyframes for a real tile.")
        plane_sets.append(tuple(int(v) for v in pixel_indices[keep].tolist()))

    state = DFROfficialTiledDecodeSchedule(
        version=C26B_SCHEDULE_VERSION,
        source_contract=(
            "frozen_tiling.TileSizeConfig+diffusion_tiling.prepare_tile_schedule+"
            "slice_stage4_tile+keyframes.planes_for_tile+group_tiles_by_temporal_slice"
        ),
        tiling_config=tiling_config,
        content_latent_shape=tuple(int(v) for v in deterministic_execution.input_video_shape),
        content_stage4_shape=tuple(int(v) for v in content_s4),
        resident_stage4_feature_shape=resident_s4,
        full_pixel_shape_bcfhw=full_shape,
        stage_min_latent_size=tuple(int(v) for v in stage_min),
        tile_min_stage4_size=tuple(int(v) for v in tile_min),
        tile_halos=tuple(tuple(int(v) for v in row) for row in tile_halos),
        pixel_scale_from_stage4=(int(pixel_scale.time), int(pixel_scale.height), int(pixel_scale.width)),
        required_temporal_overlap=int(req_t),
        required_spatial_overlap=int(req_hw),
        stage4_kernel=tuple(int(v) for v in stage_kernels[3]),
        stage4_depth=stage4_depth,
        stage5_kernel=tuple(int(v) for v in stage5_kernel),
        stage5_depth=stage5_depth,
        upsample3_stride=tuple(int(v) for v in up3_stride),
        patch_size=patch_size,
        ghost_pad_latent_frames=int(expected_ghost),
        tiles=tuple(tiles),
        group_slices=group_slices,
        tile_plane_indices=tuple(plane_sets),
        complementary_masks=bool(complementary),
        total_tiles=len(tiles),
        temporal_groups=len(groups),
    )

    unique_planes = sorted(set(state.tile_plane_indices))
    tile_preview = "; ".join(
        f"#{i}:{_tile_short(tile, state.content_stage4_shape, state.full_pixel_shape_bcfhw)}:planes={state.tile_plane_indices[i]}"
        for i, tile in enumerate(state.tiles[: min(6, len(state.tiles))])
    )
    if len(state.tiles) > 6:
        tile_preview += f"; ... +{len(state.tiles) - 6} tiles"
    report = (
        f"PASS=True; stage=U3.4b5_C26b_prepare_official_tiled_schedule; version={state.version}; "
        f"source_contract={state.source_contract}; content_latent={state.content_latent_shape}; "
        f"content_stage4={state.content_stage4_shape}; resident_stage4={state.resident_stage4_feature_shape}; "
        f"full_pixel_bcfhw={state.full_pixel_shape_bcfhw}; stage_min_latent={state.stage_min_latent_size}; "
        f"tile_min_stage4={state.tile_min_stage4_size}; tile_halos={state.tile_halos}; "
        f"pixel_scale={state.pixel_scale_from_stage4}; required_overlap=(t={state.required_temporal_overlap},hw={state.required_spatial_overlap}); "
        f"config=(frames={tile_frames}/{temporal_overlap},height={tile_height}/{spatial_overlap},width={tile_width}/{spatial_overlap}); "
        f"tiles={state.total_tiles}; temporal_groups={state.temporal_groups}; group_slices={state.group_slices}; "
        f"complementary_masks={state.complementary_masks}; distinct_plane_sets={unique_planes}; preview=[{tile_preview}]."
    )
    return state, report


def prepare_official_auto_tiled_decode_schedule(
    vae: Any,
    deterministic_execution: DFRDeterministicStageExecution,
    deterministic_context: DFRDeterministicDecoderKeyframeContext,
) -> tuple[DFROfficialTiledDecodeSchedule, str]:
    """Select the official memory-aware config, then build the C26b schedule."""
    if not isinstance(deterministic_execution, DFRDeterministicStageExecution):
        raise ValueError(
            f"Expected DFRDeterministicStageExecution, got {type(deterministic_execution).__name__}."
        )
    if not isinstance(deterministic_context, DFRDeterministicDecoderKeyframeContext):
        raise ValueError(
            f"Expected DFRDeterministicDecoderKeyframeContext, got {type(deterministic_context).__name__}."
        )
    if deterministic_execution.stage5_called:
        raise ValueError("Automatic C26b schedule requires the validated C24b state that stops before Stage 5.")

    first_stage_model = getattr(vae, "first_stage_model", None)
    decoder = getattr(first_stage_model, "decoder", None)
    if decoder is None:
        raise ValueError("Connected VAE does not expose first_stage_model.decoder.")

    stage_kernels, stage5_kernel, strides, patch_size = _runtime_decoder_geometry(decoder)
    stage4_depth = int(len(decoder.det_stages[3]))
    stage5_depth = int(len(decoder.diff_blocks))
    up3_stride = strides[3]
    tile_min = compute_tile_min_size(stage_kernels[3], stage5_kernel, up3_stride)
    tile_halos = compute_tile_halos(stage_kernels[3], stage4_depth, stage5_kernel, stage5_depth, up3_stride)
    pixel_scale = stage4_to_pixel_scale_factors(up3_stride, patch_size)

    _batch, _channels, latent_t, _latent_h, _latent_w = (
        int(v) for v in deterministic_execution.input_video_shape
    )
    time_scale = int(deterministic_context.temporal_scale_schedule[0])
    frames = (latent_t - 1) * time_scale + 1
    height, width = (int(v) for v in deterministic_context.final_pixel_spatial_shape)
    stage4_video = deterministic_execution.stage4_input_video
    stage4_channels = int(stage4_video.shape[-1])
    stage5_channels = int(getattr(decoder.diff_blocks[0].attn, "dim", 0))
    if stage5_channels < 1:
        raise ValueError("Could not resolve the loaded decoder's Stage-5 channel width.")
    element_size = 2 if stage4_video.dtype == torch.bfloat16 else int(stage4_video.element_size())
    decoder_model_bytes = _module_state_bytes(decoder)
    planning_free, current_free, total_memory, resident_decoder, resident_stage4 = _comfy_auto_tiling_free_bytes(
        vae,
        decoder,
        stage4_video,
        decoder_model_bytes=decoder_model_bytes,
    )

    config = recommended_chunked_eager_keyframe_decode_tiling_config(
        tile_halos=tile_halos,
        pixel_scale=pixel_scale,
        min_tile_size_s4=tile_min,
        patch_size=patch_size,
        height=height,
        width=width,
        num_frames=frames,
        free_bytes=planning_free,
        stage5_channels=stage5_channels,
        stage4_channels=stage4_channels,
        upsample_strides=strides,
        model_bytes=decoder_model_bytes,
        element_size=element_size,
        natten_trailing_pad_latent_frames=int(deterministic_execution.ghost_pad_latent_frames),
        out_channels=int(decoder.out_channels),
    )
    state, manual_report = prepare_official_tiled_decode_schedule(
        vae,
        deterministic_execution,
        deterministic_context,
        tile_frames=int(config.frames.tile_size),
        temporal_overlap=int(config.frames.overlap),
        tile_height=int(config.height.tile_size),
        tile_width=int(config.width.tile_size),
        spatial_overlap=int(config.height.overlap),
    )

    gib = float(1 << 30)
    report = (
        "PASS=True; stage=U3.4b5_auto_official_tiled_schedule; "
        "mode=chunked_eager; keyframes=True; joint_attention=triton_fused; "
        f"device={stage4_video.device}; current_free_gib={current_free / gib:.3f}; "
        f"total_gib={total_memory / gib:.3f}; decoder_model_gib={decoder_model_bytes / gib:.3f}; "
        f"resident_decoder_credit_gib={resident_decoder / gib:.3f}; "
        f"resident_stage4_credit_gib={resident_stage4 / gib:.3f}; planning_free_gib={planning_free / gib:.3f}; "
        f"selected=(frames={config.frames.tile_size}/{config.frames.overlap},"
        f"height={config.height.tile_size}/{config.height.overlap},"
        f"width={config.width.tile_size}/{config.width.overlap}); "
        f"tiles={state.total_tiles}; temporal_groups={state.temporal_groups}. {manual_report}"
    )
    return state, report


def _validate_schedule_against_inputs(
    schedule: DFROfficialTiledDecodeSchedule,
    deterministic_execution: DFRDeterministicStageExecution,
    deterministic_context: DFRDeterministicDecoderKeyframeContext,
) -> None:
    if not isinstance(schedule, DFROfficialTiledDecodeSchedule):
        raise ValueError(f"Expected DFROfficialTiledDecodeSchedule, got {type(schedule).__name__}.")
    if tuple(schedule.content_latent_shape) != tuple(deterministic_execution.input_video_shape):
        raise ValueError("C26b schedule was prepared for a different C24b final latent geometry.")
    if tuple(schedule.resident_stage4_feature_shape) != tuple(int(v) for v in deterministic_execution.stage4_input_video.shape):
        raise ValueError("C26b schedule was prepared for a different C24b Stage-4 feature state.")
    if tuple(deterministic_context.keyframe_pixel_frame_indices) != tuple(
        sorted(set(v for rows in schedule.tile_plane_indices for v in rows))
    ):
        # Every current generated plane must appear somewhere in the schedule; exact
        # equality is expected for the 97-frame / 32,64,96 target.
        raise ValueError("C26b schedule keyframe set does not match the connected deterministic context.")


def execute_official_tiled_decode(
    vae: Any,
    deterministic_execution: DFRDeterministicStageExecution,
    deterministic_context: DFRDeterministicDecoderKeyframeContext,
    decoder_handoff: DFRStage2DecoderHandoff,
    stage5_port: DFRExactStage5DualStreamPort,
    schedule: DFROfficialTiledDecodeSchedule,
    *,
    collect_diagnostics: bool = True,
) -> tuple[torch.Tensor, DFROfficialTiledDecodeExecution | None, str]:
    if not isinstance(decoder_handoff, DFRStage2DecoderHandoff):
        raise ValueError(f"Expected DFRStage2DecoderHandoff, got {type(decoder_handoff).__name__}.")
    if not isinstance(stage5_port, DFRExactStage5DualStreamPort):
        raise ValueError(f"Expected DFRExactStage5DualStreamPort, got {type(stage5_port).__name__}.")
    _validate_schedule_against_inputs(schedule, deterministic_execution, deterministic_context)
    if stage5_port.pathway != "official_chunked_eager_deferred_stage4_w4":
        raise ValueError(f"C26b requires C25 official chunked pathway, got {stage5_port.pathway!r}.")

    guidance = torch.empty(deterministic_execution.input_video_shape, dtype=torch.bfloat16, device="meta")
    model_management, comfy_na, comfy_kitchen = _load_comfy_vae_for_execution(vae, guidance)
    first_stage_model = vae.first_stage_model
    decoder = first_stage_model.decoder

    config = getattr(first_stage_model, "config", {}) or {}
    model_output_type = str(config.get("model_output_type", getattr(decoder, "model_output_type", "")))
    timesteps = decoder.default_inference_timesteps.detach().to(torch.float32)
    if model_output_type != "x0" or int(timesteps.numel()) != 1:
        raise ValueError(
            "C26b is locked to the validated current checkpoint (one-step x0). "
            f"Got model_output_type={model_output_type!r}, timesteps={tuple(float(v) for v in timesteps.tolist())}."
        )

    stage_kernels, stage5_kernel, strides, patch_size = _runtime_decoder_geometry(decoder)
    if tuple(stage_kernels[3]) != schedule.stage4_kernel or tuple(stage5_kernel) != schedule.stage5_kernel:
        raise ValueError("Loaded VAE kernels changed since the C26b schedule was prepared.")
    if tuple(strides[3]) != schedule.upsample3_stride or int(patch_size) != int(schedule.patch_size):
        raise ValueError("Loaded VAE Stage-4 upsample/patch geometry changed since schedule preparation.")

    device = vae.device
    dtype = vae.vae_dtype
    feat_s4 = deterministic_execution.stage4_input_video.to(device=device, dtype=dtype)
    key_s4 = deterministic_execution.stage4_input_keyframes.to(device=device, dtype=dtype)
    valid_all = deterministic_execution.stage4_input_valid.to(device=device, dtype=torch.bool)
    pixel_indices_all = torch.tensor(deterministic_context.keyframe_pixel_frame_indices, dtype=torch.long, device="cpu")
    remaining = tuple(int(v) for v in deterministic_context.temporal_scale_schedule)
    content_s4_frames = int(schedule.content_stage4_shape[0])
    full_shape = tuple(int(v) for v in schedule.full_pixel_shape_bcfhw)
    output_device = getattr(vae, "output_device", model_management.intermediate_device())
    output_dtype_fn = getattr(vae, "vae_output_dtype", None)
    output_dtype = output_dtype_fn() if callable(output_dtype_fn) else model_management.intermediate_dtype()
    expected_image_shape = (full_shape[0] * full_shape[2], full_shape[3], full_shape[4], full_shape[1])
    image = torch.empty(expected_image_shape, device=output_device, dtype=output_dtype)

    generator, rng_device_type = restore_decoder_generator(decoder_handoff)
    if collect_diagnostics:
        generator_initial = generator.get_state().detach().cpu().clone()
        expected_initial = decoder_handoff.rng_state_after_stage2_av.detach().cpu()
        generator_state_restored = torch.equal(generator_initial, expected_initial)
        if not generator_state_restored:
            raise RuntimeError("C26b reconstructed generator does not match U3.1 post-Stage2 AV state.")
        cpu_rng_before = torch.random.get_rng_state().clone()
        cuda_rng_before = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        params_before = _sample_parameter_checksum(decoder)

    groups = group_tiles_by_temporal_slice(list(schedule.tiles))
    group_slices = [slice(a, b) for a, b in schedule.group_slices]
    if len(groups) != len(group_slices):
        raise RuntimeError("C26b schedule group metadata is internally inconsistent.")

    # Frozen _decode_groups_with_keyframes: one-step x0 => group-scope x_t_init remains None.
    complementary = bool(schedule.complementary_masks)
    overlap_stub: torch.Tensor | None = None
    overlap_stub_weights: torch.Tensor | None = None
    emitted_ranges: list[tuple[int, int]] = []
    rendered_tiles = 0
    rendered_groups = 0
    blocks_total = 0
    actual_plane_sets: list[tuple[int, ...]] = []
    stage4_origins: list[float] = []
    pixel_origins: list[float] = []
    accum_dtype = torch.float16 if feat_s4.dtype == torch.bfloat16 else feat_s4.dtype
    randn_device = generator.device

    def _emit(buf: torch.Tensor, wts: torch.Tensor | None, global_start: int) -> tuple[int, int] | None:
        if global_start >= full_shape[2] or buf.shape[2] < 1:
            return None
        frames_keep = min(int(buf.shape[2]), int(full_shape[2]) - int(global_start))
        if frames_keep < 1:
            return None
        chunk = buf[:, :, :frames_keep]
        if wts is not None:
            floor = _weight_floor(wts.dtype)
            chunk = chunk / wts[:, :, :frames_keep].clamp(min=floor)
        chunk = crop_pixels_to_content(chunk.to(dtype), frames_keep, full_shape[3], full_shape[4])
        # Frozen decode_video wrapper: BFHWC then in-place [-1,1] -> [0,1].
        # Copy completed frames directly to ComfyUI's normal output device so
        # temporal groups do not accumulate as full-resolution CUDA tensors.
        bfhwc = chunk.permute(0, 2, 3, 4, 1).contiguous()
        bfhwc.add_(1).mul_(0.5).clamp_(0, 1)
        for batch_index in range(full_shape[0]):
            output_start = batch_index * full_shape[2] + global_start
            image[output_start : output_start + frames_keep].copy_(bfhwc[batch_index])
        return int(global_start), int(global_start + frames_keep)

    with model_management.cuda_device_context(device), torch.inference_mode():
        for group_index, group in enumerate(groups):
            curr_slice = group_slices[group_index]
            group_len = int(curr_slice.stop - curr_slice.start)
            group_shape = (full_shape[0], full_shape[1], group_len, full_shape[3], full_shape[4])
            buffer = torch.zeros(group_shape, device=device, dtype=accum_dtype)
            weights: torch.Tensor | None = None if complementary else torch.zeros_like(buffer)
            local_temporal_slice = slice(0, group_len)

            for tile in group:
                feat_tile, is_origin, pad_trailing, content_thw = slice_stage4_tile(
                    feat_s4, tile, content_frames=content_s4_frames
                )
                stage4_origin = tile.in_coords[1].indices(content_s4_frames)[0]
                pixel_lo, pixel_hi, _ = tile.out_coords[2].indices(full_shape[2])
                keep_cpu = planes_for_tile_official(pixel_indices_all, pixel_lo, pixel_hi - 1, clip_start_frame=0)
                if not bool(keep_cpu.any()):
                    raise RuntimeError(
                        f"Tile [{pixel_lo},{pixel_hi}) selected no keyframe planes; frozen planes_for_tile forbids this."
                    )
                tile_indices = pixel_indices_all[keep_cpu]
                if collect_diagnostics:
                    actual_plane_sets.append(tuple(int(v) for v in tile_indices.tolist()))
                    stage4_origins.append(float(stage4_origin))
                    pixel_origins.append(float(pixel_lo))

                keep_dev = keep_cpu.to(device=key_s4.device)
                # Exact KeyframeStream.select_planes(...).crop_spatial(...).
                tile_key_x = key_s4[:, keep_dev, tile.in_coords[2], tile.in_coords[3], :]
                tile_valid = valid_all[keep_dev]
                stage4_times = keyframe_clip_times_official(
                    tile_indices,
                    remaining[3],
                    0,
                    extra_origin=float(stage4_origin),
                ).to(device=device)
                tile_stream = _KeyframeStream(tile_key_x, stage4_times, tile_valid).masked()

                # Frozen CHUNKED_EAGER Stage 4: blocks only.  The last
                # pixel-shuffle is deferred into each Stage-5 block.
                context_tile, stage4_stream = _run_det_stage_blocks_with_keyframes_exact(
                    comfy_na,
                    comfy_kitchen,
                    decoder,
                    feat_tile,
                    tile_stream,
                    3,
                    num_slots=int(stage5_port.num_slots),
                )
                if pad_trailing:
                    up_t = int(strides[3][0])
                    context_tile = crop_trailing_context_natten_pad(
                        context_tile,
                        n_latent_frames=int(schedule.ghost_pad_latent_frames),
                        time_scale=int(remaining[0]) // up_t,
                        stage5_kernel_t=max(1, -(-int(schedule.stage5_kernel[0]) // up_t)),
                    )
                stage5_stream = _KeyframeStream(
                    x=stage4_stream.x,
                    times=keyframe_clip_times_official(
                        tile_indices,
                        remaining[4],
                        0,
                        extra_origin=float(pixel_lo),
                    ).to(device=device),
                    valid=stage4_stream.valid,
                ).masked()

                # Current one-step x0 branch: x_t_init is None, so draw per-tile
                # VIDEO noise first.  KEYFRAME noise is always drawn second.
                grid_t, grid_h, grid_w = stage5_grid_shape_from_stage4(
                    context_tile,
                    tuple(int(v) for v in strides[3]),
                    drop_leading_frame=bool(is_origin),
                )
                canvas_t = int(grid_t)
                canvas_h = int(grid_h) * int(patch_size)
                canvas_w = int(grid_w) * int(patch_size)
                batch = int(context_tile.shape[0])
                video_x_t = torch.randn(
                    (batch, int(decoder.out_channels), canvas_t, canvas_h, canvas_w),
                    dtype=dtype,
                    generator=generator,
                    device=randn_device,
                ).to(device)
                keyframe_x_t = torch.randn(
                    (batch, int(decoder.out_channels), int(stage5_stream.x.shape[1]), canvas_h, canvas_w),
                    dtype=dtype,
                    generator=generator,
                    device=randn_device,
                ).to(device)

                video_x, keyframe_x = build_deferred_stage5_inputs_exact(
                    comfy_na,
                    decoder,
                    context_tile,
                    stage5_stream.x,
                    video_x_t,
                    keyframe_x_t,
                    stage5_stream.valid,
                    drop_leading_frame=bool(is_origin),
                )
                t_now = timesteps.to(device=device)[0].expand(batch)
                pixel_tile, _keyframe_pred, _invalid_zero, blocks_executed = forward_stage5_chunked_step_exact(
                    comfy_na,
                    comfy_kitchen,
                    decoder,
                    video_x,
                    context_tile,
                    keyframe_x,
                    stage5_stream.x,
                    t_now,
                    stage5_stream.times,
                    stage5_stream.valid,
                    drop_leading_frame=bool(is_origin),
                    num_slots=int(stage5_port.num_slots),
                    w_chunks=OFFICIAL_STAGE5_W_CHUNKS,
                    collect_diagnostics=collect_diagnostics,
                    return_keyframes=collect_diagnostics,
                )
                blocks_total += int(blocks_executed)
                rendered_tiles += 1
                # Do not carry a completed tile's large Stage-5 hidden buffers
                # into construction of the next tile.
                del (
                    video_x,
                    keyframe_x,
                    video_x_t,
                    keyframe_x_t,
                    _keyframe_pred,
                    context_tile,
                    stage4_stream,
                    stage5_stream,
                )

                content_pixel_shape = pixel_tile_shape(full_shape, tile.out_coords)
                pixel_tile = crop_pixels_to_content(
                    pixel_tile,
                    content_pixel_shape[2],
                    content_pixel_shape[3],
                    content_pixel_shape[4],
                ).to(buffer.dtype)

                masks = tuple(m.to(device=buffer.device, dtype=torch.float32) for m in tile.masks_1d)
                local_coords = (
                    tile.out_coords[0],
                    tile.out_coords[1],
                    local_temporal_slice,
                    tile.out_coords[3],
                    tile.out_coords[4],
                )
                buffer[local_coords] += scale_by_masks_1d(pixel_tile, masks)
                if weights is not None:
                    strength = torch.ones(pixel_tile.shape, device=buffer.device, dtype=buffer.dtype)
                    weights[local_coords] += scale_by_masks_1d(strength, masks)
                    del strength
                del pixel_tile, masks

            rendered_groups += 1
            if overlap_stub is not None:
                overlap_len = int(overlap_stub.shape[2])
                if overlap_len > 0:
                    overlap_stub += buffer[:, :, :overlap_len]
                    buffer[:, :, :overlap_len] = overlap_stub
                    if not complementary:
                        assert overlap_stub_weights is not None
                        assert weights is not None
                        overlap_stub_weights += weights[:, :, :overlap_len]
                        weights[:, :, :overlap_len] = overlap_stub_weights
                overlap_stub = None
                overlap_stub_weights = None

            if group_index + 1 < len(groups):
                next_start = group_slices[group_index + 1].start
                exclusive_len = min(max(0, int(next_start) - int(curr_slice.start)), int(buffer.shape[2]))
                emitted = _emit(
                    buffer[:, :, :exclusive_len],
                    None if weights is None else weights[:, :, :exclusive_len],
                    int(curr_slice.start),
                )
                if emitted is not None:
                    emitted_ranges.append(emitted)
                overlap_stub = buffer[:, :, exclusive_len:].clone()
                if not complementary:
                    assert weights is not None
                    overlap_stub_weights = weights[:, :, exclusive_len:].clone()
                del buffer, weights
            else:
                emitted = _emit(buffer, weights, int(curr_slice.start))
                if emitted is not None:
                    emitted_ranges.append(emitted)
                del buffer, weights

    if not emitted_ranges:
        raise RuntimeError("C26b produced no emitted pixel chunks.")
    expected_start = 0
    for emitted_start, emitted_end in emitted_ranges:
        if emitted_start != expected_start or emitted_end <= emitted_start:
            raise RuntimeError(f"C26b emitted non-contiguous frame range [{emitted_start},{emitted_end}).")
        expected_start = emitted_end
    if expected_start != full_shape[2]:
        raise RuntimeError(f"C26b emitted {expected_start} frames, expected {full_shape[2]}.")
    if tuple(int(v) for v in image.shape) != expected_image_shape:
        raise RuntimeError(
            f"C26b assembled IMAGE shape {tuple(image.shape)} != exact target {expected_image_shape}."
        )

    if not collect_diagnostics:
        return image, None, ""

    params_after = _sample_parameter_checksum(decoder)
    cpu_rng_after = torch.random.get_rng_state().clone()
    cuda_rng_after = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    generator_final = generator.get_state().detach().cpu().clone()

    state = DFROfficialTiledDecodeExecution(
        version=C26B_EXECUTION_VERSION,
        source_contract=(
            "frozen_DiffusionVideoDecoder._decode_groups_with_keyframes+"
            "_decode_temporal_group_isolated_with_keyframes+_decode_one_tile_with_keyframes+"
            "forward_stage_4_with_keyframes(deferred)+forward_diff_step_deferred_with_keyframes+"
            "chunked.context+chunked.attn+chunked.mlp+keyframes.planes_for_tile+diffusion_tiling"
        ),
        execution_mode="official_CHUNKED_EAGER_w4_explicit_TileSizeConfig_tiled_one_step_x0_keyframe_decode",
        device=str(device),
        dtype=str(dtype),
        model_output_type=model_output_type,
        inference_timesteps=tuple(float(v) for v in timesteps.tolist()),
        output_image_shape=tuple(int(v) for v in image.shape),
        output_pixel_shape_bcfhw=full_shape,
        total_tiles=int(schedule.total_tiles),
        rendered_tiles=int(rendered_tiles),
        temporal_groups=int(schedule.temporal_groups),
        rendered_groups=int(rendered_groups),
        complementary_masks=bool(complementary),
        stage5_block_count=int(len(decoder.diff_blocks)),
        stage5_blocks_executed_total=int(blocks_total),
        min_planes_per_tile=min(len(v) for v in actual_plane_sets),
        max_planes_per_tile=max(len(v) for v in actual_plane_sets),
        tile_plane_indices=tuple(actual_plane_sets),
        stage4_time_origins=tuple(stage4_origins),
        pixel_time_origins=tuple(pixel_origins),
        generator_state_restored=bool(generator_state_restored),
        generator_state_advanced=not torch.equal(generator_initial, generator_final),
        generator_initial_state=generator_initial,
        generator_final_state=generator_final,
        rng_device_type=rng_device_type,
        cpu_rng_unchanged=torch.equal(cpu_rng_before, cpu_rng_after),
        cuda_rng_unchanged=(True if device.type != "cuda" else torch.equal(cuda_rng_before, cuda_rng_after)),
        decoder_parameters_unchanged=params_before == params_after,
        decoder_mutated=False,
    )
    report = execution_report(state, schedule)
    return image, state, report


def validate_official_tiled_decode(
    execution: DFROfficialTiledDecodeExecution,
    schedule: DFROfficialTiledDecodeSchedule,
    decoder_handoff: DFRStage2DecoderHandoff,
) -> dict[str, Any]:
    if not isinstance(execution, DFROfficialTiledDecodeExecution):
        raise ValueError(f"Expected DFROfficialTiledDecodeExecution, got {type(execution).__name__}.")
    if not isinstance(schedule, DFROfficialTiledDecodeSchedule):
        raise ValueError(f"Expected DFROfficialTiledDecodeSchedule, got {type(schedule).__name__}.")
    if not isinstance(decoder_handoff, DFRStage2DecoderHandoff):
        raise ValueError(f"Expected DFRStage2DecoderHandoff, got {type(decoder_handoff).__name__}.")

    expected_image = (
        schedule.full_pixel_shape_bcfhw[0] * schedule.full_pixel_shape_bcfhw[2],
        schedule.full_pixel_shape_bcfhw[3],
        schedule.full_pixel_shape_bcfhw[4],
        schedule.full_pixel_shape_bcfhw[1],
    )
    geometry_error = 0.0 if tuple(execution.output_image_shape) == tuple(expected_image) else 1.0
    tile_error = 0.0 if execution.rendered_tiles == schedule.total_tiles else 1.0
    group_error = 0.0 if execution.rendered_groups == schedule.temporal_groups else 1.0
    blocks_expected = schedule.total_tiles * execution.stage5_block_count
    block_error = 0.0 if execution.stage5_blocks_executed_total == blocks_expected else 1.0
    plane_error = 0.0 if tuple(execution.tile_plane_indices) == tuple(schedule.tile_plane_indices) else 1.0
    rng_initial_error = 0.0 if torch.equal(
        execution.generator_initial_state.detach().cpu(), decoder_handoff.rng_state_after_stage2_av.detach().cpu()
    ) else 1.0
    rng_advance_error = 0.0 if execution.generator_state_advanced else 1.0
    invariants_error = 0.0 if (
        execution.generator_state_restored
        and execution.cpu_rng_unchanged
        and execution.cuda_rng_unchanged
        and execution.decoder_parameters_unchanged
        and not execution.decoder_mutated
    ) else 1.0
    passed = all(
        v == 0.0
        for v in (
            geometry_error,
            tile_error,
            group_error,
            block_error,
            plane_error,
            rng_initial_error,
            rng_advance_error,
            invariants_error,
        )
    )
    return {
        "passed": bool(passed),
        "geometry_error": geometry_error,
        "tile_error": tile_error,
        "group_error": group_error,
        "block_error": block_error,
        "plane_error": plane_error,
        "rng_initial_error": rng_initial_error,
        "rng_advance_error": rng_advance_error,
        "invariants_error": invariants_error,
        "output_image_shape": execution.output_image_shape,
        "expected_image_shape": expected_image,
        "rendered_tiles": execution.rendered_tiles,
        "total_tiles": schedule.total_tiles,
        "rendered_groups": execution.rendered_groups,
        "temporal_groups": schedule.temporal_groups,
        "stage5_block_count": execution.stage5_block_count,
        "stage5_blocks_executed_total": execution.stage5_blocks_executed_total,
        "expected_stage5_block_calls": blocks_expected,
        "min_planes_per_tile": execution.min_planes_per_tile,
        "max_planes_per_tile": execution.max_planes_per_tile,
        "generator_state_restored": execution.generator_state_restored,
        "generator_state_advanced": execution.generator_state_advanced,
        "cpu_rng_unchanged": execution.cpu_rng_unchanged,
        "cuda_rng_unchanged": execution.cuda_rng_unchanged,
        "decoder_parameters_unchanged": execution.decoder_parameters_unchanged,
        "complementary_masks": execution.complementary_masks,
        "group_slices": schedule.group_slices,
        "tile_plane_indices": execution.tile_plane_indices,
    }


def execution_report(state: DFROfficialTiledDecodeExecution, schedule: DFROfficialTiledDecodeSchedule) -> str:
    return (
        f"PASS=True; stage=U3.4b5_C26b_execute_official_tiled_decode; version={state.version}; "
        f"source_contract={state.source_contract}; mode={state.execution_mode}; device={state.device}; dtype={state.dtype}; "
        f"checkpoint={state.model_output_type}/{state.inference_timesteps}; output_image={state.output_image_shape}; "
        f"output_bcfhw={state.output_pixel_shape_bcfhw}; "
        f"tiles={state.rendered_tiles}/{state.total_tiles}; groups={state.rendered_groups}/{state.temporal_groups}; "
        f"group_slices={schedule.group_slices}; complementary_masks={state.complementary_masks}; "
        f"planes_per_tile={state.min_planes_per_tile}..{state.max_planes_per_tile}; "
        f"stage5_blocks={state.stage5_blocks_executed_total} ({state.stage5_block_count}/tile); "
        f"generator_state_restored={state.generator_state_restored}; generator_state_advanced={state.generator_state_advanced}; "
        f"cpu_rng_unchanged={state.cpu_rng_unchanged}; cuda_rng_unchanged={state.cuda_rng_unchanged}; "
        f"decoder_parameters_unchanged={state.decoder_parameters_unchanged}; decoder_mutated=False."
    )
