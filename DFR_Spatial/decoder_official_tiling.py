"""C26b / U3.4b5: exact frozen DiffVAE tiling geometry used by the keyframe decoder.

This is a narrow native-Comfy port of the relevant functions from the frozen
LTX-core ``tiling.py`` and ``model/video_vae/diffusion_tiling.py`` sources.
It includes the official auto-VRAM recommender specialized to the validated
CHUNKED_EAGER keyframe decode, while C26b can still accept an explicit
``TileSizeConfig`` unchanged.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple, Sequence

import torch


@dataclass(frozen=True)
class SpatioTemporalScaleFactors:
    time: int
    height: int
    width: int


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors(time=8, height=32, width=32)

_DEFAULT_ELEMENT_SIZE = 2
_ACCUMULATOR_CHANNELS = 3
_MIN_MODEL_BYTES_FLOOR = 1 << 30
_CHUNKED_EAGER_KEYFRAME_STAGE5_MEM_COEF = 5.0
_CHUNKED_EAGER_KEYFRAME_RESERVE_BYTES = 1 << 30

# Peak of the official SDR encode sink, expressed with doubled half-channel
# counts so the arithmetic remains integral.
_CONVERT_PEAK_UV_CHANNELS_X2 = 5
_PACK_PEAK_UV_CHANNELS_X2 = 4
_PACK_PEAK_UINT8_BYTES_X2 = 3


@dataclass(frozen=True)
class DimensionInterval:
    start: int
    end: int
    left_ramp: int
    right_ramp: int


@dataclass(frozen=True)
class DimensionIntervals:
    intervals: list[DimensionInterval]


SplitOperation = Callable[[int], DimensionIntervals]


def default_split_operation(length: int) -> DimensionIntervals:
    return DimensionIntervals(intervals=[DimensionInterval(start=0, end=length, left_ramp=0, right_ramp=0)])


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


def untiled_mask_1d() -> torch.Tensor:
    return torch.ones(1)


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> torch.Tensor:
    if length <= 0:
        raise ValueError("Mask length must be positive.")
    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))
    mask = torch.ones(length)
    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = torch.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask[:ramp_left] *= fade_in
    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, steps=ramp_right + 2)[1:-1]
        mask[-ramp_right:] *= fade_out
    return mask.clamp_(0, 1)


def _grow_last_tile_to_min(intervals: list[DimensionInterval], min_tile_size: int) -> list[DimensionInterval]:
    if len(intervals) <= 1:
        return list(intervals)
    last = intervals[-1]
    if last.end - last.start >= min_tile_size:
        return list(intervals)
    new_start = last.end - min_tile_size
    prev = intervals[-2]
    new_overlap = prev.end - new_start
    return [
        *intervals[:-2],
        replace(prev, right_ramp=new_overlap),
        replace(last, start=new_start, left_ramp=new_overlap),
    ]


def _validate_tile_intervals(intervals: list[DimensionInterval], *, dim_size: int, min_tile_size: int) -> None:
    if not intervals or intervals[0].start != 0 or intervals[-1].end != dim_size:
        raise ValueError(f"tiles must cover [0, {dim_size})")
    for i, iv in enumerate(intervals):
        length = iv.end - iv.start
        if length < min_tile_size:
            raise ValueError(f"tile {i} length {length} is below min_tile_size={min_tile_size}")
        if iv.left_ramp < 0 or iv.right_ramp < 0 or iv.left_ramp > length or iv.right_ramp > length:
            raise ValueError(
                f"tile {i} has invalid ramps: left={iv.left_ramp}, right={iv.right_ramp}, length={length}"
            )
        if i == 0:
            continue
        overlap = intervals[i - 1].end - iv.start
        if overlap < 0 or intervals[i - 1].right_ramp != overlap or iv.left_ramp != overlap:
            raise ValueError(f"tiles {i - 1}/{i}: ramp/overlap mismatch (overlap={overlap})")


def split_by_size(size: int, overlap: int, min_tile_size: int | None = None) -> SplitOperation:
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got overlap={overlap}, size={size}")
    if min_tile_size is not None and min_tile_size < 1:
        raise ValueError(f"min_tile_size must be >= 1, got {min_tile_size}")

    def split(dimension_size: int) -> DimensionIntervals:
        if min_tile_size is not None and dimension_size < min_tile_size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        intervals = [
            DimensionInterval(start=0, end=size, left_ramp=0, right_ramp=overlap),
            *(
                DimensionInterval(
                    start=i * (size - overlap),
                    end=i * (size - overlap) + size,
                    left_ramp=overlap,
                    right_ramp=overlap,
                )
                for i in range(1, amount - 1)
            ),
            DimensionInterval(
                start=(amount - 1) * (size - overlap), end=dimension_size, left_ramp=overlap, right_ramp=0
            ),
        ]
        if min_tile_size is not None:
            intervals = _grow_last_tile_to_min(intervals, min_tile_size)
            _validate_tile_intervals(intervals, dim_size=dimension_size, min_tile_size=min_tile_size)
        return DimensionIntervals(intervals=intervals)

    return split


@dataclass(frozen=True)
class DimensionSizeConfig:
    tile_size: int = 0
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.tile_size < 0:
            raise ValueError(f"tile_size must be >= 0, got {self.tile_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")
        if self.tile_size == 0:
            if self.overlap != 0:
                raise ValueError("untiled axis (tile_size=0) must have overlap=0")
            return
        if self.overlap >= self.tile_size:
            raise ValueError(f"Overlap must be less than tile size, got {self.overlap} and {self.tile_size}")

    def is_tiled(self) -> bool:
        return self.tile_size > 0


def _validate_size_axis(cfg: DimensionSizeConfig, factor: int, axis_name: str) -> None:
    if not cfg.is_tiled():
        return
    min_size = 2 * factor
    if cfg.tile_size < min_size:
        raise ValueError(f"{axis_name}.tile_size must be at least {min_size}, got {cfg.tile_size}")
    if cfg.tile_size % factor != 0:
        raise ValueError(f"{axis_name}.tile_size must be divisible by {factor}, got {cfg.tile_size}")
    if cfg.overlap % factor != 0:
        raise ValueError(f"{axis_name}.overlap must be divisible by {factor}, got {cfg.overlap}")


@dataclass(frozen=True)
class TileSizeConfig:
    frames: DimensionSizeConfig = DimensionSizeConfig()
    height: DimensionSizeConfig = DimensionSizeConfig()
    width: DimensionSizeConfig = DimensionSizeConfig()

    def to_splitters(
        self,
        scale_factors: SpatioTemporalScaleFactors,
        min_tile_size: tuple[int, int, int] | None = None,
        *,
        causal_temporal: bool = True,
    ) -> tuple[SplitOperation, SplitOperation, SplitOperation]:
        min_t = min_h = min_w = None
        if min_tile_size is not None:
            min_t, min_h, min_w = min_tile_size

        def enable_size_axis(
            factor: int,
            axis_min: int | None,
            cfg: DimensionSizeConfig,
            axis_name: str,
            *,
            temporal: bool,
        ) -> SplitOperation:
            if not cfg.is_tiled():
                return DEFAULT_SPLIT_OPERATION
            _validate_size_axis(cfg, factor, axis_name)
            size = cfg.tile_size // factor
            overlap = cfg.overlap // factor
            lower_threshold = max(2, overlap + 1)
            tile = max(lower_threshold, size)
            # prepare_tile_schedule passes causal_temporal=False.  The frozen source
            # branches to split_temporal_causal only when True; C26b never takes it.
            if temporal and causal_temporal:
                raise ValueError("C26b prepare_tile_schedule must use causal_temporal=False exactly.")
            return split_by_size(tile, overlap, min_tile_size=axis_min)

        return (
            enable_size_axis(scale_factors.time, min_t, self.frames, "frames", temporal=True),
            enable_size_axis(scale_factors.height, min_h, self.height, "height", temporal=False),
            enable_size_axis(scale_factors.width, min_w, self.width, "width", temporal=False),
        )


class Tile(NamedTuple):
    in_coords: tuple[slice, ...]
    out_coords: tuple[slice, ...]
    masks_1d: tuple[torch.Tensor, ...]

    @property
    def blend_mask(self) -> torch.Tensor:
        num_dims = len(self.out_coords)
        per_dimension_masks: list[torch.Tensor] = []
        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.view(*view_shape))
        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask
        return combined_mask


def scale_by_masks_1d(x: torch.Tensor, masks_1d: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(masks_1d) != x.ndim:
        raise ValueError(f"masks_1d length {len(masks_1d)} != x.ndim {x.ndim}")
    out = x
    for axis, mask in enumerate(masks_1d):
        view_shape = [1] * x.ndim
        view_shape[axis] = -1
        out = out * mask.reshape(*view_shape)
    return out


def masks_are_complementary(
    tiles: Sequence[Tile],
    full_shape: Sequence[int],
    *,
    atol: float = 1e-5,
) -> bool:
    if not tiles:
        return True
    ndim = len(full_shape)
    for tile in tiles:
        if len(tile.out_coords) != ndim or len(tile.masks_1d) != ndim:
            raise ValueError(
                f"Tile out_coords/masks_1d rank {len(tile.out_coords)}/{len(tile.masks_1d)} != full_shape rank {ndim}"
            )
    for axis, length in enumerate(full_shape):
        acc = torch.zeros(length, dtype=torch.float32, device="cpu")
        seen: set[tuple[int | None, int | None]] = set()
        for tile in tiles:
            sl = tile.out_coords[axis]
            key = (sl.start, sl.stop)
            if key in seen:
                continue
            seen.add(key)
            acc[sl] += tile.masks_1d[axis].detach().float().cpu()
        if not torch.allclose(acc, torch.ones(length, dtype=torch.float32), atol=atol, rtol=0.0):
            return False
    return True


def group_tiles_by_temporal_slice(tiles: list[Tile]) -> list[list[Tile]]:
    if not tiles:
        return []
    groups = []
    current_slice = tiles[0].out_coords[2]
    current_group = []
    for tile in tiles:
        tile_slice = tile.out_coords[2]
        if tile_slice == current_slice:
            current_group.append(tile)
        else:
            groups.append(current_group)
            current_slice = tile_slice
            current_group = [tile]
    if current_group:
        groups.append(current_group)
    return groups


def _validate_overlap(
    tiling_config: TileSizeConfig,
    *,
    min_overlap_frames: int,
    min_overlap_pixels: int,
) -> None:
    for axis_name, cfg, recommended, unit in (
        ("frames", tiling_config.frames, min_overlap_frames, "frames"),
        ("height", tiling_config.height, min_overlap_pixels, "px"),
        ("width", tiling_config.width, min_overlap_pixels, "px"),
    ):
        if cfg.is_tiled() and cfg.overlap < recommended:
            raise ValueError(f"{axis_name} overlap {cfg.overlap} {unit} is below the required {recommended} {unit}.")


def _propagate_interval_through_upsample_hops(
    interval: DimensionInterval,
    strides: Sequence[int],
    causal: bool,
) -> DimensionInterval:
    x = interval
    for stride in strides:
        if stride < 1:
            raise ValueError(f"upsample stride must be >= 1, got {stride}")
        start = x.start * stride
        end = x.end * stride
        left_ramp = x.left_ramp * stride
        right_ramp = x.right_ramp * stride
        if causal and stride == 2:
            end -= 1
            if x.start != 0:
                start -= 1
        x = DimensionInterval(start=start, end=end, left_ramp=left_ramp, right_ramp=right_ramp)
    return x


def stage4_to_pixel_scale_factors(
    upsample_stride: tuple[int, int, int],
    patch_size: int,
) -> SpatioTemporalScaleFactors:
    st, sh, sw = upsample_stride
    return SpatioTemporalScaleFactors(time=st, height=sh * patch_size, width=sw * patch_size)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def recommended_pixel_overlaps(
    tile_halos: tuple[tuple[int, int, int], tuple[int, int, int]],
    pixel_scale: SpatioTemporalScaleFactors,
) -> tuple[int, int]:
    def dominant(axis: int) -> int:
        return max(tile_halos[i][axis] for i in range(len(tile_halos)))

    overlap_t = _round_up(dominant(0) * pixel_scale.time, 8)
    halo_hw = max(dominant(1), dominant(2))
    overlap_hw = _round_up(halo_hw * pixel_scale.height, 32)
    return overlap_t, overlap_hw


def _frames_per_yuv_gemm(height: int, width: int) -> int:
    """Frozen ``ltx_core.color.yuv.frames_per_yuv_gemm`` arithmetic."""
    int32_max = 2**31 - 1
    if height <= 0 or width <= 0:
        return int32_max
    limit = int32_max // int(height) // int(width)
    return 1 if limit <= 1 else limit - 1


def _max_emitted_frames(*, num_frames: int, tile_frames: int, overlap_frames: int) -> int:
    if tile_frames >= num_frames:
        return num_frames
    stride = tile_frames - overlap_frames
    if stride <= 0:
        return tile_frames
    n_tiles = 1 + -(-(num_frames - tile_frames) // stride)
    last_group = num_frames - (n_tiles - 1) * stride
    return min(tile_frames, max(stride, last_group + 1))


def _emit_convert_bytes(
    *,
    tile_frames: int,
    height: int,
    width: int,
    out_channels: int,
    element_size: int,
) -> int:
    """Official downstream RGB-to-YUV/pack peak used by auto-tiling."""
    frames = int(tile_frames)
    gemm_frames = min(frames, _frames_per_yuv_gemm(height, width))
    resident_yuv_frames = frames if gemm_frames == frames else 0
    yuv_channels_x2 = 2 * int(out_channels)
    convert_peak_x2 = element_size * (
        yuv_channels_x2 * gemm_frames + _CONVERT_PEAK_UV_CHANNELS_X2 * frames
    )
    pack_peak_x2 = (
        element_size * (yuv_channels_x2 * resident_yuv_frames + _PACK_PEAK_UV_CHANNELS_X2 * frames)
        + _PACK_PEAK_UINT8_BYTES_X2 * frames
    )
    return int(height) * int(width) * max(convert_peak_x2, pack_peak_x2) // 2


def _stage4_feature_bytes(
    *,
    height: int,
    width: int,
    num_frames: int,
    upsample_strides: Sequence[tuple[int, int, int]],
    stage4_channels: int,
    element_size: int,
    natten_trailing_pad_latent_frames: int,
) -> int:
    """Resident stages-1-to-3 output size from the official budget model."""
    latent_t = (int(num_frames) - 1) // VIDEO_SCALE_FACTORS.time + 1
    latent_h = int(height) // VIDEO_SCALE_FACTORS.height
    latent_w = int(width) // VIDEO_SCALE_FACTORS.width
    s4_t, s4_h, s4_w = stage4_thw_from_latent(
        upsample_strides[:3],
        latent_t + int(natten_trailing_pad_latent_frames),
        latent_h,
        latent_w,
        drop_leading_frame=True,
    )
    return int(s4_t) * int(s4_h) * int(s4_w) * int(stage4_channels) * int(element_size)


def _stage5_tokens_for_pixel_tile(
    tile_frames: int,
    tile_height: int,
    tile_width: int,
    *,
    patch_size: int,
) -> int:
    h5 = max(1, tile_height // patch_size)
    w5 = max(1, tile_width // patch_size)
    return tile_frames * h5 * w5


def _axis_candidates(length: int, overlap: int, min_size: int, multiple: int) -> list[tuple[int, int]]:
    """``(tile_size, tile_count)`` on the exact official candidate grid."""
    out: list[tuple[int, int]] = []
    max_size = max(_round_up(length, multiple), min_size)
    for size in range(min_size, max_size + multiple, multiple):
        if size <= overlap:
            continue
        n = len(split_by_size(size, overlap)(length).intervals)
        out.append((size, n))
    return out


def _volumetric_overlap_waste(
    *,
    num_frames: int,
    height: int,
    width: int,
    tile_frames: int,
    tile_height: int,
    tile_width: int,
    n_t: int,
    n_h: int,
    n_w: int,
) -> float:
    processed = n_t * n_h * n_w * tile_frames * tile_height * tile_width
    unique = max(1, num_frames * height * width)
    return processed / unique


def recommended_chunked_eager_keyframe_decode_tiling_config(  # noqa: PLR0913
    *,
    tile_halos: tuple[tuple[int, int, int], tuple[int, int, int]],
    pixel_scale: SpatioTemporalScaleFactors,
    min_tile_size_s4: tuple[int, int, int],
    patch_size: int,
    height: int,
    width: int,
    num_frames: int,
    free_bytes: int,
    stage5_channels: int,
    stage4_channels: int,
    upsample_strides: Sequence[tuple[int, int, int]],
    model_bytes: int = 0,
    element_size: int = _DEFAULT_ELEMENT_SIZE,
    natten_trailing_pad_latent_frames: int = 0,
    out_channels: int = _ACCUMULATOR_CHANNELS,
) -> TileSizeConfig:
    """Official auto-tiler specialized to this port's validated decode mode.

    The full LTX helper supports four execution modes.  The current Comfy parity
    executor is deliberately fixed to keyframe-aware CHUNKED_EAGER with fused
    Triton joint attention, so its official coefficient is 5 and its reserve is
    1 GiB.  Candidate enumeration, peak estimate and tie-breaking are otherwise
    the frozen ``recommended_decode_tiling_config`` implementation.
    """
    if height < 1 or width < 1 or num_frames < 1:
        raise ValueError(f"height/width/num_frames must be >= 1, got {height}x{width}x{num_frames}")
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    if stage5_channels < 1 or stage4_channels < 1:
        raise ValueError(
            f"stage4_channels/stage5_channels must be >= 1, got {stage4_channels}/{stage5_channels}"
        )
    if out_channels < 1 or element_size < 1:
        raise ValueError(f"out_channels/element_size must be >= 1, got {out_channels}/{element_size}")

    overlap_t, overlap_hw = recommended_pixel_overlaps(tile_halos, pixel_scale)
    ft, fh, fw = pixel_scale.time, pixel_scale.height, pixel_scale.width
    step_t = math.lcm(ft, VIDEO_SCALE_FACTORS.time)
    step_h = math.lcm(fh, VIDEO_SCALE_FACTORS.height)
    step_w = math.lcm(fw, VIDEO_SCALE_FACTORS.width)
    min_t_px = _round_up(
        max(2 * ft, 2 * overlap_t, _round_up(min_tile_size_s4[0] * ft, ft), 16),
        step_t,
    )
    min_h_px = _round_up(
        max(2 * fh, 2 * overlap_hw, _round_up(min_tile_size_s4[1] * fh, fh), 64),
        step_h,
    )
    min_w_px = _round_up(
        max(2 * fw, 2 * overlap_hw, _round_up(min_tile_size_s4[2] * fw, fw), 64),
        step_w,
    )

    model_cost = max(int(model_bytes), _MIN_MODEL_BYTES_FLOOR)
    s4_feat_bytes = _stage4_feature_bytes(
        height=height,
        width=width,
        num_frames=num_frames,
        upsample_strides=upsample_strides,
        stage4_channels=stage4_channels,
        element_size=element_size,
        natten_trailing_pad_latent_frames=natten_trailing_pad_latent_frames,
    )
    usable = max(
        0,
        int(free_bytes) - model_cost - _CHUNKED_EAGER_KEYFRAME_RESERVE_BYTES - s4_feat_bytes,
    )
    s5_bytes_per_token = max(
        1.0,
        float(stage5_channels) * float(element_size) * _CHUNKED_EAGER_KEYFRAME_STAGE5_MEM_COEF,
    )
    acc_bytes_per_pixel = int(out_channels) * int(element_size)

    t_cands = _axis_candidates(num_frames, overlap_t, min_t_px, step_t)
    h_cands = _axis_candidates(height, overlap_hw, min_h_px, step_h)
    w_cands = _axis_candidates(width, overlap_hw, min_w_px, step_w)
    scored: list[tuple[float, int, int, int, int, int]] = []
    for tile_t, n_t in t_cands:
        acc_frames = 2 * int(tile_t)
        acc_bytes = acc_frames * int(height) * int(width) * acc_bytes_per_pixel
        downstream_bytes = _emit_convert_bytes(
            tile_frames=_max_emitted_frames(
                num_frames=num_frames,
                tile_frames=tile_t,
                overlap_frames=overlap_t,
            ),
            height=height,
            width=width,
            out_channels=out_channels,
            element_size=element_size,
        )
        if acc_bytes + downstream_bytes >= usable:
            continue
        max_s5_tokens = int((usable - acc_bytes - downstream_bytes) // s5_bytes_per_token)
        for tile_h, n_h in h_cands:
            for tile_w, n_w in w_cands:
                if (
                    _stage5_tokens_for_pixel_tile(
                        tile_t,
                        tile_h,
                        tile_w,
                        patch_size=patch_size,
                    )
                    > max_s5_tokens
                ):
                    continue
                waste = _volumetric_overlap_waste(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    tile_frames=tile_t,
                    tile_height=tile_h,
                    tile_width=tile_w,
                    n_t=n_t,
                    n_h=n_h,
                    n_w=n_w,
                )
                scored.append((waste, -tile_t * tile_h * tile_w, n_t * n_h * n_w, tile_t, tile_h, tile_w))

    if not scored:
        raise ValueError(
            "Cannot fit a DiffVAE decode tile under the memory budget: "
            f"min tile ~{min_t_px}f x {min_h_px}x{min_w_px}px "
            f"(overlaps T={overlap_t}, HW={overlap_hw}), mode=chunked_eager, "
            f"keyframes=True, coef={_CHUNKED_EAGER_KEYFRAME_STAGE5_MEM_COEF:g}, "
            f"stage5_channels={stage5_channels}, stage4_feature_bytes={s4_feat_bytes}, "
            f"usable_bytes={usable}. Reduce resolution, reduce num_frames, or free GPU memory."
        )

    scored.sort()
    _waste, _volume, _tile_count, tile_t, tile_h, tile_w = scored[0]
    return TileSizeConfig(
        frames=DimensionSizeConfig(tile_size=tile_t, overlap=overlap_t),
        height=DimensionSizeConfig(tile_size=tile_h, overlap=overlap_hw),
        width=DimensionSizeConfig(tile_size=tile_w, overlap=overlap_hw),
    )


def prepare_tile_schedule(
    stage4_shape_bcthw: torch.Size,
    tiling_config: TileSizeConfig | None,
    *,
    upsample3_stride: tuple[int, int, int],
    patch_size: int,
    min_tile_size: tuple[int, int, int],
    tile_halos: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> list[Tile]:
    pixel_scale = stage4_to_pixel_scale_factors(upsample3_stride, patch_size)
    if tiling_config is None:
        return [
            Tile(
                in_coords=(slice(None), slice(None), slice(None), slice(None), slice(None)),
                out_coords=(slice(None), slice(None), slice(None), slice(None), slice(None)),
                masks_1d=(
                    untiled_mask_1d(), untiled_mask_1d(), untiled_mask_1d(), untiled_mask_1d(), untiled_mask_1d()
                ),
            )
        ]

    overlap_t, overlap_hw = recommended_pixel_overlaps(tile_halos, pixel_scale)
    _validate_overlap(tiling_config, min_overlap_frames=overlap_t, min_overlap_pixels=overlap_hw)
    t_split, h_split, w_split = tiling_config.to_splitters(
        pixel_scale, min_tile_size=min_tile_size, causal_temporal=False
    )
    st, sh, sw = upsample3_stride

    def axis_specs(
        split_op: SplitOperation,
        dim_len: int,
        stride_component: int,
        *,
        propagate_causal: bool,
        apply_patch: bool,
    ) -> list[tuple[slice, slice, torch.Tensor]]:
        if split_op is DEFAULT_SPLIT_OPERATION:
            return [(slice(None), slice(None), untiled_mask_1d())]
        intervals = split_op(dim_len).intervals
        specs = []
        for iv in intervals:
            stage5 = _propagate_interval_through_upsample_hops(iv, [stride_component], propagate_causal)
            if apply_patch:
                pixel = _propagate_interval_through_upsample_hops(stage5, [patch_size], causal=False)
            else:
                pixel = stage5
            mask_pixel = compute_trapezoidal_mask_1d(
                pixel.end - pixel.start, pixel.left_ramp, pixel.right_ramp, left_starts_from_0=False
            )
            specs.append((slice(iv.start, iv.end), slice(pixel.start, pixel.end), mask_pixel))
        return specs

    t_specs = axis_specs(t_split, stage4_shape_bcthw[2], st, propagate_causal=True, apply_patch=False)
    h_specs = axis_specs(h_split, stage4_shape_bcthw[3], sh, propagate_causal=False, apply_patch=True)
    w_specs = axis_specs(w_split, stage4_shape_bcthw[4], sw, propagate_causal=False, apply_patch=True)

    tiles: list[Tile] = []
    for t_spec, h_spec, w_spec in itertools.product(t_specs, h_specs, w_specs):
        t_s4, t_px, t_mask = t_spec
        h_s4, h_px, h_mask = h_spec
        w_s4, w_px, w_mask = w_spec
        tiles.append(
            Tile(
                in_coords=(slice(None), t_s4, h_s4, w_s4, slice(None)),
                out_coords=(slice(None), slice(None), t_px, h_px, w_px),
                masks_1d=(untiled_mask_1d(), untiled_mask_1d(), t_mask, h_mask, w_mask),
            )
        )
    return tiles


def slice_stage4_tile(
    feat_s4: torch.Tensor,
    tile: Tile,
    *,
    content_frames: int,
) -> tuple[torch.Tensor, bool, bool, tuple[int, int, int]]:
    is_origin = tile.in_coords[1].start in (0, None)
    _, stop, _ = tile.in_coords[1].indices(content_frames)
    pad_trailing = stop == content_frames
    _b, t_coord, h_coord, w_coord, _c = tile.in_coords
    t0, t1, _ = t_coord.indices(content_frames)
    h0, h1, _ = h_coord.indices(feat_s4.shape[2])
    w0, w1, _ = w_coord.indices(feat_s4.shape[3])
    content_thw = (t1 - t0, h1 - h0, w1 - w0)
    if pad_trailing:
        t1 = feat_s4.shape[1]
    feat_tile = feat_s4[:, t0:t1, h_coord, w_coord, :]
    return feat_tile, is_origin, pad_trailing, content_thw


@dataclass(frozen=True)
class AxisPad:
    before: int
    after: int


def resize_axis(
    x: torch.Tensor,
    dim: int,
    size: int,
    *,
    mode: str,
) -> tuple[torch.Tensor, AxisPad]:
    if size < 1:
        raise ValueError(f"resize_axis target size must be >= 1, got {size}")
    if dim < 0:
        dim += x.ndim
    if not 0 <= dim < x.ndim:
        raise ValueError(f"dim {dim} out of range for rank-{x.ndim} tensor")
    length = x.shape[dim]
    if length == size:
        return x, AxisPad(0, 0)
    if length < size:
        need = size - length
        if mode == "repeat_last":
            last = x.narrow(dim, length - 1, 1)
            expand_shape = list(x.shape)
            expand_shape[dim] = need
            pad = last.expand(expand_shape)
            return torch.cat([x, pad], dim=dim), AxisPad(0, need)
        if mode != "symmetric":
            raise ValueError(f"unsupported resize mode {mode!r}")
        before = need // 2
        after = need - before
        first = x.narrow(dim, 0, 1)
        last = x.narrow(dim, length - 1, 1)
        parts: list[torch.Tensor] = []
        if before:
            expand_shape = list(x.shape)
            expand_shape[dim] = before
            parts.append(first.expand(expand_shape))
        parts.append(x)
        if after:
            expand_shape = list(x.shape)
            expand_shape[dim] = after
            parts.append(last.expand(expand_shape))
        return torch.cat(parts, dim=dim), AxisPad(before, after)
    need = length - size
    if mode == "repeat_last":
        return x.narrow(dim, 0, size).contiguous(), AxisPad(0, need)
    if mode != "symmetric":
        raise ValueError(f"unsupported resize mode {mode!r}")
    before = need // 2
    after = need - before
    return x.narrow(dim, before, size).contiguous(), AxisPad(before, after)


def crop_pixels_to_content(
    pixels: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    *,
    h_pad: AxisPad | None = None,
    w_pad: AxisPad | None = None,
    spatial_scale: tuple[int, int] = (1, 1),
) -> torch.Tensor:
    x, _ = resize_axis(pixels, 2, frames, mode="repeat_last")
    scale_h, scale_w = spatial_scale
    if h_pad is not None:
        before = h_pad.before * scale_h
        if before + height > x.shape[3]:
            raise ValueError(f"H crop out of range: before={before}, height={height}, got {x.shape[3]}")
        x = x.narrow(3, before, height).contiguous()
    else:
        x, _ = resize_axis(x, 3, height, mode="symmetric")
    if w_pad is not None:
        before = w_pad.before * scale_w
        if before + width > x.shape[4]:
            raise ValueError(f"W crop out of range: before={before}, width={width}, got {x.shape[4]}")
        x = x.narrow(4, before, width).contiguous()
    else:
        x, _ = resize_axis(x, 4, width, mode="symmetric")
    return x


def stage5_pixel_shape_from_stage4(
    stage4_t: int,
    stage4_h: int,
    stage4_w: int,
    *,
    upsample_stride: tuple[int, int, int],
    patch_size: int,
    stage5_kernel_t: int,
    drop_leading_frame: bool,
    pad_trailing: bool,
) -> tuple[int, int, int]:
    st, sh, sw = upsample_stride
    frames = stage4_t * st - 1 if drop_leading_frame and st == 2 else stage4_t * st
    if pad_trailing:
        frames = max(frames, stage5_kernel_t)
    return frames, stage4_h * sh * patch_size, stage4_w * sw * patch_size


def crop_trailing_context_natten_pad(
    context: torch.Tensor,
    *,
    n_latent_frames: int,
    time_scale: int,
    stage5_kernel_t: int,
) -> torch.Tensor:
    if n_latent_frames <= 0:
        return context
    ghost = n_latent_frames * time_scale
    content_t = max(context.shape[1] - ghost, 1)
    keep = min(context.shape[1], max(content_t, stage5_kernel_t))
    cropped, _ = resize_axis(context, 1, keep, mode="repeat_last")
    return cropped


def stage4_thw_from_latent(
    upsample_strides: Sequence[tuple[int, int, int]],
    latent_t: int,
    latent_h: int,
    latent_w: int,
    *,
    drop_leading_frame: bool = True,
) -> tuple[int, int, int]:
    t, h, w = latent_t, latent_h, latent_w
    for st, sh, sw in upsample_strides[:3]:
        t, h, w = t * st, h * sh, w * sw
        if st == 2 and drop_leading_frame:
            t -= 1
    return t, h, w


def compute_tile_min_size(
    stage4_kernel: tuple[int, int, int],
    stage5_kernel: tuple[int, int, int],
    upsample3_stride: tuple[int, int, int],
) -> tuple[int, int, int]:
    return tuple(max(stage4_kernel[a], -(-stage5_kernel[a] // upsample3_stride[a])) for a in range(3))


def compute_tile_halos(
    stage4_kernel: tuple[int, int, int],
    stage4_depth: int,
    stage5_kernel: tuple[int, int, int],
    stage5_depth: int,
    upsample3_stride: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    halo4 = tuple(stage4_depth * (stage4_kernel[a] // 2) for a in range(3))
    halo5 = tuple(-(-(stage5_depth * (stage5_kernel[a] // 2)) // upsample3_stride[a]) for a in range(3))
    return halo4, halo5  # type: ignore[return-value]


def _cumulative_upsample_strides(
    upsamples: Sequence[tuple[tuple[int, int, int], int]],
) -> list[tuple[int, int, int]]:
    cumulative = [(1, 1, 1)]
    t, h, w = 1, 1, 1
    for stride, _ in upsamples:
        t, h, w = t * stride[0], h * stride[1], w * stride[2]
        cumulative.append((t, h, w))
    return cumulative


def all_stages_min_tile_size(
    stage_kernels: Sequence[tuple[int, int, int]],
    upsamples: Sequence[tuple[tuple[int, int, int], int]],
    stage5_kernel: tuple[int, int, int],
) -> tuple[int, int, int]:
    cumulative = _cumulative_upsample_strides(upsamples)
    mins = [1, 1, 1]
    for stage_i in range(len(upsamples)):
        strides = cumulative[stage_i]
        for axis in range(3):
            mins[axis] = max(mins[axis], -(-stage_kernels[stage_i][axis] // strides[axis]))
    strides5 = cumulative[len(upsamples)]
    for axis in range(3):
        mins[axis] = max(mins[axis], -(-stage5_kernel[axis] // strides5[axis]))
    return (mins[0], mins[1], mins[2])


def pixel_tile_shape(full_shape: tuple[int, ...], out_coords: tuple[slice, ...]) -> tuple[int, ...]:
    dims: list[int] = []
    for size, coord in zip(full_shape, out_coords, strict=True):
        start, stop, step = coord.indices(size)
        dims.append(len(range(start, stop, step)))
    return tuple(dims)


def _weight_floor(dtype: torch.dtype) -> float:
    return max(1e-8, torch.finfo(dtype).tiny)
