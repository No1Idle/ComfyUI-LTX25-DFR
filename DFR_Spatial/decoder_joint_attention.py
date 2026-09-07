"""U3.4b2 joint video/keyframe neighborhood attention.

This module ports the official LTX DiffVAE joint-attention semantics and its
runtime backend choice: the official Triton kernel on CUDA when Triton imports,
and the existing limited-workspace eager implementation otherwise.

The algorithm carries two streams through a single softmax per query:
- video query: local 3-D video neighborhood + spatial windows from the nearest
  keyframe planes;
- keyframe query: its own plane's spatial neighborhood + spatial windows from
  the nearest video frames.

Keyframe/video slot selection is ranked by ``(|dt|, candidate_index)`` with two
cross-stream slots, matching the official source.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any

import torch
import torch.nn.functional as F

from .decoder_keyframe_substrate import DFRDecoderKeyframeSubstrate
from .decoder_keyframe_checkpoint import DFRDecoderKeyframeCheckpointWeights


KEYFRAME_CONTEXT_SLOTS = 2
_DEAD = -1.0e4
_HEAD_DIM_ALIGN = 8
DEFAULT_BRICK_QUERIES = 64
DEFAULT_BRICK_DEPTH = 4
DEFAULT_WORKSPACE_BYTES = 256 * 1024**2
_STAGING_FACTOR_FUSED = 4.75
_STAGING_FACTOR_MATERIALIZED = 22.1
JOINT_ATTENTION_VERSION = 2
JOINT_ATTENTION_BACKEND = "official_auto_joint_triton_cuda_eager_fallback"


@dataclass(frozen=True)
class DFRJointDecoderAttention:
    version: int
    num_slots: int
    source_backend: str
    checkpoint_has_type_emb: bool
    type_emb_shape: tuple[int, ...]
    keyframe_channels: int
    channels_match_expected: bool


def _nearest_slots(
    query_times: torch.Tensor,
    candidate_times: torch.Tensor,
    candidate_valid: torch.Tensor | None,
    num_slots: int,
) -> torch.Tensor:
    """Candidate indices ranked by ``(|dt|, index)``, ``-1`` for empty slots."""
    distances = (query_times[:, None] - candidate_times[None, :]).abs().to(torch.float32)
    if candidate_valid is not None:
        distances = distances.masked_fill(~candidate_valid[None, :], float("inf"))
    order = torch.argsort(distances, dim=-1, stable=True)
    take = min(num_slots, candidate_times.shape[0])
    chosen = order[:, :take]
    finite = torch.gather(distances, 1, chosen).isfinite()
    chosen = torch.where(finite, chosen, torch.full_like(chosen, -1))
    if take < num_slots:
        pad = torch.full(
            (chosen.shape[0], num_slots - take),
            -1,
            dtype=chosen.dtype,
            device=chosen.device,
        )
        chosen = torch.cat([chosen, pad], dim=1)
    return chosen


def video_keyframe_slots(
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    video_length: int,
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> torch.Tensor:
    query = torch.arange(video_length, dtype=torch.float32, device=keyframe_times.device)
    return _nearest_slots(query, keyframe_times.to(torch.float32), keyframe_valid, num_slots)


def keyframe_video_slots(
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    video_length: int,
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> torch.Tensor:
    candidates = torch.arange(video_length, dtype=torch.float32, device=keyframe_times.device)
    slots = _nearest_slots(keyframe_times.to(torch.float32), candidates, None, num_slots)
    return torch.where(keyframe_valid[:, None], slots, torch.full_like(slots, -1))


def sdpa_materializes_scores(device: torch.device) -> bool:
    return device.type != "cuda"


def staging_factor(device: torch.device) -> float:
    return _STAGING_FACTOR_MATERIALIZED if sdpa_materializes_scores(device) else _STAGING_FACTOR_FUSED


def _key_channels(head_dim: int) -> int:
    return -(-(head_dim + 1) // _HEAD_DIM_ALIGN) * _HEAD_DIM_ALIGN


def _window(kernel: int) -> tuple[int, int]:
    lo = kernel // 2
    return lo, kernel - lo - 1


def pick_brick(
    time: int,
    height: int,
    width: int,
    target: int = DEFAULT_BRICK_QUERIES,
    depth: int = DEFAULT_BRICK_DEPTH,
) -> tuple[int, int, int]:
    side = max(1, round(math.sqrt(target)))
    return min(depth, time), min(side, height), min(side, width)


class _Geometry:
    def __init__(
        self,
        height: int,
        width: int,
        kernel: tuple[int, int, int],
        brick: tuple[int, int, int],
    ) -> None:
        kernel_t, kernel_h, kernel_w = kernel
        lo_h, hi_h = _window(kernel_h)
        lo_w, hi_w = _window(kernel_w)
        self.height, self.width = height, width
        self.brick = brick
        self.kernel = kernel
        self.grid = (-(-height // brick[1]), -(-width // brick[2]))
        self.span_t = brick[0] + kernel_t - 1
        self.span = (brick[1] + kernel_h - 1, brick[2] + kernel_w - 1)
        self.pad_h = (lo_h, hi_h + self.grid[0] * brick[1] - height)
        self.pad_w = (lo_w, hi_w + self.grid[1] * brick[2] - width)
        self.pad_t = _window(kernel_t)
        self.queries = brick[0] * brick[1] * brick[2]
        self.footprint = self.span[0] * self.span[1]
        self.padded_height = height + sum(self.pad_h)
        self.padded_width = width + sum(self.pad_w)

    def row_extent(self, rows: int) -> int:
        return (rows - 1) * self.brick[1] + self.span[0]


class _Schedule:
    def __init__(
        self,
        geometry: _Geometry,
        blocks: int,
        heads: int,
        head_dim: int,
        axis_bricks: int,
        element_size: int,
        workspace_bytes: int,
        factor: float,
    ) -> None:
        channels = _key_channels(head_dim)
        keys = blocks * geometry.footprint
        pair_bytes = geometry.grid[1] * heads * keys * (channels + head_dim) * element_size
        pairs = max(1, int(workspace_bytes / max(pair_bytes * factor, 1.0)))
        if pairs >= geometry.grid[0]:
            self.group_axis = min(axis_bricks, max(1, pairs // geometry.grid[0]))
            self.group_rows = geometry.grid[0]
        else:
            self.group_axis = 1
            self.group_rows = pairs
        staged = (
            geometry.padded_height
            * geometry.padded_width
            * heads
            * (channels + head_dim)
            * element_size
        )
        per_axis_brick = staged * geometry.brick[0]
        self.stage_axis = min(
            axis_bricks,
            max(self.group_axis, workspace_bytes // max(per_axis_brick, 1)),
        )


def _banded(queries: int, span: int, kernel: int, device: torch.device) -> torch.Tensor:
    key = torch.arange(span, device=device)[None, :]
    query = torch.arange(queries, device=device)[:, None]
    return (key >= query) & (key < query + kernel)


def _joint_mask(geometry: _Geometry, num_slots: int, device: torch.device) -> torch.Tensor:
    brick_t, brick_h, brick_w = geometry.brick
    kernel_t, kernel_h, kernel_w = geometry.kernel
    spatial = (
        _banded(brick_h, geometry.span[0], kernel_h, device)[:, None, :, None]
        & _banded(brick_w, geometry.span[1], kernel_w, device)[None, :, None, :]
    ).reshape(brick_h * brick_w, geometry.footprint)
    temporal = _banded(brick_t, geometry.span_t, kernel_t, device)
    video = (temporal[:, None, :, None] & spatial[None, :, None, :]).reshape(
        geometry.queries, geometry.span_t * geometry.footprint
    )
    planes = (
        spatial[None, :, None, :]
        .expand(brick_t, brick_h * brick_w, num_slots, geometry.footprint)
        .reshape(geometry.queries, num_slots * geometry.footprint)
    )
    return torch.cat([video, planes], dim=1)[None, None].contiguous()


def _stage(
    x: torch.Tensor,
    geometry: _Geometry,
    pad_t: tuple[int, int],
    *,
    with_bias_channel: bool,
) -> torch.Tensor:
    batch, axis, height, width, heads, head_dim = x.shape
    channels = _key_channels(head_dim) if with_bias_channel else head_dim
    out = x.new_zeros(
        (
            batch,
            heads,
            axis + sum(pad_t),
            geometry.padded_height,
            geometry.padded_width,
            channels,
        )
    )
    if with_bias_channel:
        out[..., head_dim] = _DEAD
    live = out[
        :,
        :,
        pad_t[0] : pad_t[0] + axis,
        geometry.pad_h[0] : geometry.pad_h[0] + height,
        geometry.pad_w[0] : geometry.pad_w[0] + width,
    ]
    live[..., :head_dim] = x.permute(0, 4, 1, 2, 3, 5)
    if with_bias_channel:
        live[..., head_dim] = 0.0
    return out


def _slabs(
    staged: torch.Tensor,
    geometry: _Geometry,
    bricks: int,
    rows: int,
    blocks: int,
    *,
    group_stride: int,
) -> torch.Tensor:
    batch, heads = staged.shape[0], staged.shape[1]
    stride_b, stride_nh, stride_a, stride_h, stride_w, _ = staged.stride()
    return staged.as_strided(
        (
            batch,
            bricks,
            rows,
            geometry.grid[1],
            heads,
            blocks,
            *geometry.span,
            staged.shape[-1],
        ),
        (
            stride_b,
            group_stride * stride_a,
            geometry.brick[1] * stride_h,
            geometry.brick[2] * stride_w,
            stride_nh,
            stride_a,
            stride_h,
            stride_w,
            1,
        ),
    )


def _query_bricks(x: torch.Tensor, geometry: _Geometry, bricks: int, rows: int) -> torch.Tensor:
    batch, axis, height, width, heads, head_dim = x.shape
    brick_t, brick_h, brick_w = geometry.brick
    pad_t = bricks * brick_t - axis
    pad_h = rows * brick_h - height
    pad_w = geometry.grid[1] * brick_w - width
    if pad_t or pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_w, 0, pad_h, 0, pad_t))
    bricked = (
        x.reshape(
            batch,
            bricks,
            brick_t,
            rows,
            brick_h,
            geometry.grid[1],
            brick_w,
            heads,
            head_dim,
        )
        .permute(0, 1, 3, 5, 7, 2, 4, 6, 8)
        .reshape(batch * bricks * rows * geometry.grid[1], heads, geometry.queries, head_dim)
    )
    out = bricked.new_zeros((*bricked.shape[:-1], _key_channels(head_dim)))
    out[..., :head_dim] = bricked
    out[..., head_dim] = 1.0
    return out


def _unbrick(
    attended: torch.Tensor,
    geometry: _Geometry,
    batch: int,
    bricks: int,
    rows: int,
    extent: tuple[int, int],
) -> torch.Tensor:
    brick_t, brick_h, brick_w = geometry.brick
    heads, head_dim = attended.shape[1], attended.shape[3]
    plane = (
        attended.reshape(
            batch,
            bricks,
            rows,
            geometry.grid[1],
            heads,
            brick_t,
            brick_h,
            brick_w,
            head_dim,
        )
        .permute(0, 1, 5, 2, 6, 3, 7, 4, 8)
        .reshape(
            batch,
            bricks * brick_t,
            rows * brick_h,
            geometry.grid[1] * brick_w,
            heads,
            head_dim,
        )
    )
    return plane[:, : extent[0], : extent[1], : geometry.width]


def _with_null(slots: torch.Tensor, null_index: int) -> torch.Tensor:
    return torch.where(slots < 0, torch.full_like(slots, null_index), slots)


def _append_null(
    keys: torch.Tensor,
    values: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (keys.shape[0], keys.shape[1], 1, *keys.shape[3:])
    null_key = keys.new_zeros(shape)
    null_key[..., head_dim] = _DEAD
    null_value = values.new_zeros((*shape[:-1], values.shape[-1]))
    return torch.cat([keys, null_key], dim=2), torch.cat([values, null_value], dim=2)


def _slot_runs(slots: torch.Tensor) -> list[tuple[int, int]]:
    rows = slots.tolist()
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(rows)):
        if rows[index] != rows[start]:
            runs.append((start, index))
            start = index
    runs.append((start, len(rows)))
    return runs


def _attend_group(
    query_slice: torch.Tensor,
    key_views: tuple[torch.Tensor, ...],
    value_views: tuple[torch.Tensor, ...],
    geometry: _Geometry,
    shape: tuple[int, int],
    mask: torch.Tensor,
) -> torch.Tensor:
    bricks, rows = shape
    batch = query_slice.shape[0]
    heads, head_dim = query_slice.shape[4], query_slice.shape[5]
    blocks = sum(view.shape[5] for view in key_views)
    channels = _key_channels(head_dim)
    keys = query_slice.new_empty(
        (
            batch,
            bricks,
            rows,
            geometry.grid[1],
            heads,
            blocks,
            *geometry.span,
            channels,
        )
    )
    values = query_slice.new_empty(
        (
            batch,
            bricks,
            rows,
            geometry.grid[1],
            heads,
            blocks,
            *geometry.span,
            head_dim,
        )
    )
    start = 0
    for key_view, value_view in zip(key_views, value_views, strict=True):
        stop = start + key_view.shape[5]
        keys[:, :, :, :, :, start:stop].copy_(key_view)
        values[:, :, :, :, :, start:stop].copy_(value_view)
        start = stop
    count = batch * bricks * rows * geometry.grid[1]
    attended = F.scaled_dot_product_attention(
        _query_bricks(query_slice, geometry, bricks, rows),
        keys.view(count, heads, blocks * geometry.footprint, channels),
        values.view(count, heads, blocks * geometry.footprint, head_dim),
        attn_mask=mask,
        scale=1.0,
    )
    return _unbrick(
        attended,
        geometry,
        batch,
        bricks,
        rows,
        (query_slice.shape[1], query_slice.shape[2]),
    )


def _row_groups(geometry: _Geometry, schedule: _Schedule) -> list[tuple[int, int, slice, slice]]:
    brick_h = geometry.brick[1]
    groups = []
    for row in range(0, geometry.grid[0], schedule.group_rows):
        rows = min(schedule.group_rows, geometry.grid[0] - row)
        groups.append(
            (
                row,
                rows,
                slice(row * brick_h, row * brick_h + geometry.row_extent(rows)),
                slice(row * brick_h, min((row + rows) * brick_h, geometry.height)),
            )
        )
    return groups


def _video_query_pass(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    slots: torch.Tensor,
    geometry: _Geometry,
    workspace_bytes: int,
    factor: float,
) -> torch.Tensor:
    time, heads, head_dim = q.shape[1], q.shape[4], q.shape[5]
    brick_t = geometry.brick[0]
    lo_t, hi_t = geometry.pad_t
    num_slots = slots.shape[1]
    blocks = geometry.span_t + num_slots

    plane_keys, plane_values = _append_null(
        _stage(keyframe_k, geometry, (0, 0), with_bias_channel=True),
        _stage(keyframe_v, geometry, (0, 0), with_bias_channel=False),
        head_dim,
    )
    slot_table = _with_null(slots, keyframe_k.shape[1])
    mask = _joint_mask(geometry, num_slots, q.device)
    schedule = _Schedule(
        geometry,
        blocks,
        heads,
        head_dim,
        -(-time // brick_t),
        q.element_size(),
        workspace_bytes,
        factor,
    )
    rows_groups = _row_groups(geometry, schedule)

    out = torch.empty_like(q)
    for run_start, run_stop in _slot_runs(slot_table):
        planes = plane_keys.index_select(2, slot_table[run_start])
        plane_vals = plane_values.index_select(2, slot_table[run_start])
        run_bricks = -(-(run_stop - run_start) // brick_t)
        for staged_brick in range(0, run_bricks, schedule.stage_axis):
            staged_bricks = min(schedule.stage_axis, run_bricks - staged_brick)
            first = run_start + staged_brick * brick_t
            last = first + staged_bricks * brick_t
            source = slice(max(0, first - lo_t), min(time, last + hi_t))
            pad_t = (max(0, lo_t - first), max(0, last + hi_t - time))
            window_keys = _stage(k[:, source], geometry, pad_t, with_bias_channel=True)
            window_values = _stage(v[:, source], geometry, pad_t, with_bias_channel=False)

            for brick in range(staged_brick, staged_brick + staged_bricks, schedule.group_axis):
                count = min(schedule.group_axis, staged_brick + staged_bricks - brick)
                start = run_start + brick * brick_t
                stop = min(start + count * brick_t, run_stop)
                offset = (brick - staged_brick) * brick_t
                for _, rows, key_rows, out_rows in rows_groups:
                    tile = _attend_group(
                        q[:, start:stop, out_rows],
                        (
                            _slabs(
                                window_keys[:, :, offset:, key_rows],
                                geometry,
                                count,
                                rows,
                                geometry.span_t,
                                group_stride=brick_t,
                            ),
                            _slabs(
                                planes[:, :, :, key_rows],
                                geometry,
                                count,
                                rows,
                                num_slots,
                                group_stride=0,
                            ),
                        ),
                        (
                            _slabs(
                                window_values[:, :, offset:, key_rows],
                                geometry,
                                count,
                                rows,
                                geometry.span_t,
                                group_stride=brick_t,
                            ),
                            _slabs(
                                plane_vals[:, :, :, key_rows],
                                geometry,
                                count,
                                rows,
                                num_slots,
                                group_stride=0,
                            ),
                        ),
                        geometry,
                        (count, rows),
                        mask,
                    )
                    out[:, start:stop, out_rows] = tile
    return out


def _keyframe_query_pass(
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slots: torch.Tensor,
    keyframe_valid: torch.Tensor,
    geometry: _Geometry,
    workspace_bytes: int,
    factor: float,
) -> torch.Tensor:
    planes_total, heads, head_dim = keyframe_q.shape[1], keyframe_q.shape[4], keyframe_q.shape[5]
    num_slots = slots.shape[1]
    blocks = 1 + num_slots
    time = k.shape[1]
    flat = _Geometry(
        geometry.height,
        geometry.width,
        (1, *geometry.kernel[1:]),
        (1, *geometry.brick[1:]),
    )

    wanted, inverse = torch.unique(_with_null(slots, time).reshape(-1), return_inverse=True)
    frame_keys = _stage(
        k.index_select(1, wanted.clamp(max=time - 1)),
        flat,
        (0, 0),
        with_bias_channel=True,
    )
    frame_values = _stage(
        v.index_select(1, wanted.clamp(max=time - 1)),
        flat,
        (0, 0),
        with_bias_channel=False,
    )
    frame_keys[:, :, wanted == time, ..., head_dim] = _DEAD
    own_keys = _stage(keyframe_k, flat, (0, 0), with_bias_channel=True)
    own_values = _stage(keyframe_v, flat, (0, 0), with_bias_channel=False)
    own_keys[:, :, ~keyframe_valid, ..., head_dim] = _DEAD
    slot_table = inverse.reshape(planes_total, num_slots)
    mask = _joint_mask(flat, num_slots, keyframe_q.device)
    schedule = _Schedule(
        flat,
        blocks,
        heads,
        head_dim,
        planes_total,
        keyframe_q.element_size(),
        workspace_bytes,
        factor,
    )
    rows_groups = _row_groups(flat, schedule)

    out = torch.empty_like(keyframe_q)
    for start in range(0, planes_total, schedule.group_axis):
        stop = min(start + schedule.group_axis, planes_total)
        count = stop - start
        picked = slot_table[start:stop].reshape(-1)
        frames = frame_keys.index_select(2, picked)
        frame_vals = frame_values.index_select(2, picked)
        for _, rows, key_rows, out_rows in rows_groups:
            tile = _attend_group(
                keyframe_q[:, start:stop, out_rows],
                (
                    _slabs(own_keys[:, :, start:, key_rows], flat, count, rows, 1, group_stride=1),
                    _slabs(frames[:, :, :, key_rows], flat, count, rows, num_slots, group_stride=num_slots),
                ),
                (
                    _slabs(own_values[:, :, start:, key_rows], flat, count, rows, 1, group_stride=1),
                    _slabs(frame_vals[:, :, :, key_rows], flat, count, rows, num_slots, group_stride=num_slots),
                ),
                flat,
                (count, rows),
                mask,
            )
            out[:, start:stop, out_rows] = tile
    return out * keyframe_valid[None, :, None, None, None, None]


def joint_na3d_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: tuple[int, int, int],
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
    brick: tuple[int, int, int] | None = None,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    factor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official fallback semantics for joint video/keyframe 3-D neighborhood attention."""
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"video q/k/v shapes must match; got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
    if keyframe_q.shape != keyframe_k.shape or keyframe_q.shape != keyframe_v.shape:
        raise ValueError(
            "keyframe q/k/v shapes must match; "
            f"got q={tuple(keyframe_q.shape)}, k={tuple(keyframe_k.shape)}, v={tuple(keyframe_v.shape)}"
        )
    if q.ndim != 6 or keyframe_q.ndim != 6:
        raise ValueError("joint_na3d expects video/keyframe tensors shaped [B,T/P,H,W,NH,HD].")
    if q.shape[0] != keyframe_q.shape[0] or q.shape[2:4] != keyframe_q.shape[2:4] or q.shape[4:] != keyframe_q.shape[4:]:
        raise ValueError("video and keyframe streams must share B/H/W/NH/HD geometry.")
    if keyframe_times.shape != (keyframe_q.shape[1],):
        raise ValueError("keyframe_times must be [P].")
    if keyframe_valid.shape != (keyframe_q.shape[1],) or keyframe_valid.dtype != torch.bool:
        raise ValueError("keyframe_valid must be bool [P].")
    if num_slots < 1:
        raise ValueError("num_slots must be >= 1.")

    time, height, width = q.shape[1], q.shape[2], q.shape[3]
    video_slots = video_keyframe_slots(keyframe_times, keyframe_valid, time, num_slots)
    keyframe_slots = keyframe_video_slots(keyframe_times, keyframe_valid, time, num_slots)
    geometry = _Geometry(
        height,
        width,
        kernel_size,
        brick if brick is not None else pick_brick(time, height, width),
    )
    if factor is None:
        factor = staging_factor(q.device)
    return (
        _video_query_pass(
            q,
            k,
            v,
            keyframe_k,
            keyframe_v,
            video_slots,
            geometry,
            workspace_bytes,
            factor,
        ),
        _keyframe_query_pass(
            keyframe_q,
            keyframe_k,
            keyframe_v,
            k,
            v,
            keyframe_slots,
            keyframe_valid,
            geometry,
            workspace_bytes,
            factor,
        ),
    )


@lru_cache(maxsize=1)
def triton_joint_attention_available() -> bool:
    """Whether the official joint Triton backend can be selected on this host."""
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401, PLC0415
    except (ImportError, OSError):
        return False
    return True


def _joint_dtypes(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Match the official wrapper: RoPE-promoted Q/K return to V's dtype."""
    if q.dtype != v.dtype or k.dtype != v.dtype:
        q, k = q.to(dtype=v.dtype), k.to(dtype=v.dtype)
    if keyframe_q.dtype != keyframe_v.dtype or keyframe_k.dtype != keyframe_v.dtype:
        keyframe_q = keyframe_q.to(dtype=keyframe_v.dtype)
        keyframe_k = keyframe_k.to(dtype=keyframe_v.dtype)
    return q, k, v, keyframe_q, keyframe_k, keyframe_v


def selected_joint_attention_backend(
    q: torch.Tensor,
    *,
    brick: tuple[int, int, int] | None = None,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    factor: float | None = None,
) -> str:
    """Return the per-call backend selected by the official tensor-device rule."""
    use_default_eager_controls = (
        brick is None and int(workspace_bytes) == DEFAULT_WORKSPACE_BYTES and factor is None
    )
    if q.is_cuda and use_default_eager_controls and triton_joint_attention_available():
        return "triton"
    return "eager"


def joint_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: tuple[int, int, int],
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
    brick: tuple[int, int, int] | None = None,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
    factor: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official auto joint attention: Triton on CUDA tensors, eager otherwise."""
    q, k, v, keyframe_q, keyframe_k, keyframe_v = _joint_dtypes(
        q, k, v, keyframe_q, keyframe_k, keyframe_v
    )
    if selected_joint_attention_backend(
        q,
        brick=brick,
        workspace_bytes=workspace_bytes,
        factor=factor,
    ) == "triton":
        from .decoder_joint_attention_triton import joint_na3d as triton_joint_na3d  # noqa: PLC0415

        return triton_joint_na3d(
            q,
            k,
            v,
            keyframe_q,
            keyframe_k,
            keyframe_v,
            keyframe_times,
            keyframe_valid,
            kernel_size,
            num_slots,
        )
    return joint_na3d_eager(
        q,
        k,
        v,
        keyframe_q,
        keyframe_k,
        keyframe_v,
        keyframe_times,
        keyframe_valid,
        kernel_size,
        num_slots,
        brick,
        workspace_bytes,
        factor,
    )


def _brute_force_joint_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keyframe_q: torch.Tensor,
    keyframe_k: torch.Tensor,
    keyframe_v: torch.Tensor,
    keyframe_times: torch.Tensor,
    keyframe_valid: torch.Tensor,
    kernel_size: tuple[int, int, int],
    num_slots: int = KEYFRAME_CONTEXT_SLOTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small-tensor semantic oracle used only by the U3.4b2 validator."""
    bsz, time, height, width, heads, head_dim = q.shape
    planes = keyframe_q.shape[1]
    kt, kh, kw = kernel_size
    lo_t, lo_h, lo_w = kt // 2, kh // 2, kw // 2
    hi_t, hi_h, hi_w = kt - lo_t - 1, kh - lo_h - 1, kw - lo_w - 1
    vslots = video_keyframe_slots(keyframe_times, keyframe_valid, time, num_slots)
    kslots = keyframe_video_slots(keyframe_times, keyframe_valid, time, num_slots)
    out_v = torch.zeros_like(q)
    out_kf = torch.zeros_like(keyframe_q)

    def attend(query: torch.Tensor, keys: list[torch.Tensor], vals: list[torch.Tensor]) -> torch.Tensor:
        # query [B,NH,HD], stacked keys/vals [B,NH,NK,HD]. q is already pre-scaled.
        kk = torch.stack(keys, dim=2)
        vv = torch.stack(vals, dim=2)
        scores = (query[:, :, None, :] * kk).sum(dim=-1)
        probs = torch.softmax(scores, dim=-1)
        return (probs[..., None] * vv).sum(dim=2)

    for t in range(time):
        for h in range(height):
            for w in range(width):
                keys: list[torch.Tensor] = []
                vals: list[torch.Tensor] = []
                for tt in range(max(0, t - lo_t), min(time, t + hi_t + 1)):
                    for hh in range(max(0, h - lo_h), min(height, h + hi_h + 1)):
                        for ww in range(max(0, w - lo_w), min(width, w + hi_w + 1)):
                            keys.append(k[:, tt, hh, ww])
                            vals.append(v[:, tt, hh, ww])
                for plane in vslots[t].tolist():
                    if plane < 0:
                        continue
                    for hh in range(max(0, h - lo_h), min(height, h + hi_h + 1)):
                        for ww in range(max(0, w - lo_w), min(width, w + hi_w + 1)):
                            keys.append(keyframe_k[:, plane, hh, ww])
                            vals.append(keyframe_v[:, plane, hh, ww])
                out_v[:, t, h, w] = attend(q[:, t, h, w], keys, vals)

    for plane in range(planes):
        if not bool(keyframe_valid[plane]):
            continue
        for h in range(height):
            for w in range(width):
                keys = []
                vals = []
                # Own keyframe plane only.
                for hh in range(max(0, h - lo_h), min(height, h + hi_h + 1)):
                    for ww in range(max(0, w - lo_w), min(width, w + hi_w + 1)):
                        keys.append(keyframe_k[:, plane, hh, ww])
                        vals.append(keyframe_v[:, plane, hh, ww])
                for frame in kslots[plane].tolist():
                    if frame < 0:
                        continue
                    for hh in range(max(0, h - lo_h), min(height, h + hi_h + 1)):
                        for ww in range(max(0, w - lo_w), min(width, w + hi_w + 1)):
                            keys.append(k[:, frame, hh, ww])
                            vals.append(v[:, frame, hh, ww])
                out_kf[:, plane, h, w] = attend(keyframe_q[:, plane, h, w], keys, vals)
    return out_v, out_kf


def prepare_joint_decoder_attention(
    decoder_keyframe_substrate: DFRDecoderKeyframeSubstrate,
    decoder_keyframe_checkpoint_weights: DFRDecoderKeyframeCheckpointWeights,
) -> tuple[DFRJointDecoderAttention, str]:
    if not isinstance(decoder_keyframe_substrate, DFRDecoderKeyframeSubstrate):
        raise ValueError(
            f"Expected DFRDecoderKeyframeSubstrate, got {type(decoder_keyframe_substrate).__name__}."
        )
    if not isinstance(decoder_keyframe_checkpoint_weights, DFRDecoderKeyframeCheckpointWeights):
        raise ValueError(
            "Expected DFRDecoderKeyframeCheckpointWeights, got "
            f"{type(decoder_keyframe_checkpoint_weights).__name__}."
        )
    if not decoder_keyframe_substrate.keyframe_channels_match_expected:
        raise ValueError("U3.4b2 requires the validated 128-channel keyframe substrate.")
    if not decoder_keyframe_checkpoint_weights.checkpoint_has_type_emb:
        raise ValueError("U3.4b2 requires the real trained decoder.type_emb from the LTX-2.5 checkpoint.")
    shape = tuple(int(x) for x in decoder_keyframe_checkpoint_weights.type_emb.shape)
    if shape != (decoder_keyframe_substrate.expected_keyframe_channels,):
        raise ValueError(
            "Checkpoint type_emb shape does not match the validated keyframe channel width: "
            f"shape={shape}, channels={decoder_keyframe_substrate.expected_keyframe_channels}."
        )
    state = DFRJointDecoderAttention(
        version=JOINT_ATTENTION_VERSION,
        num_slots=KEYFRAME_CONTEXT_SLOTS,
        source_backend=JOINT_ATTENTION_BACKEND,
        checkpoint_has_type_emb=True,
        type_emb_shape=shape,
        keyframe_channels=int(decoder_keyframe_substrate.keyframe_channels),
        channels_match_expected=True,
    )
    report = (
        f"PASS=True; stage=U3.4b2_prepare_joint_attention; version={state.version}; "
        f"backend={state.source_backend}; num_slots={state.num_slots}; "
        f"checkpoint_has_type_emb={state.checkpoint_has_type_emb}; type_emb_shape={state.type_emb_shape}; "
        f"keyframe_channels={state.keyframe_channels}; channels_match_expected={state.channels_match_expected}; "
        "decoder_mutated=False; decoder_called=False."
    )
    return state, report


def validate_joint_decoder_attention(
    joint_attention_state: DFRJointDecoderAttention,
    test_device: str = "cuda",
) -> dict[str, Any]:
    if not isinstance(joint_attention_state, DFRJointDecoderAttention):
        raise ValueError(f"Expected DFRJointDecoderAttention, got {type(joint_attention_state).__name__}.")
    requested = str(test_device).strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        requested = "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported test_device={test_device!r}.")
    device = torch.device(requested)

    gen = torch.Generator(device=device).manual_seed(3402)
    shape_v = (1, 4, 3, 3, 2, 8)
    shape_k = (1, 3, 3, 3, 2, 8)
    q = torch.randn(shape_v, generator=gen, device=device, dtype=torch.float32) / math.sqrt(shape_v[-1])
    k = torch.randn(shape_v, generator=gen, device=device, dtype=torch.float32)
    v = torch.randn(shape_v, generator=gen, device=device, dtype=torch.float32)
    kq = torch.randn(shape_k, generator=gen, device=device, dtype=torch.float32) / math.sqrt(shape_k[-1])
    kk = torch.randn(shape_k, generator=gen, device=device, dtype=torch.float32)
    kv = torch.randn(shape_k, generator=gen, device=device, dtype=torch.float32)
    times = torch.tensor([0.25, 2.0, 5.0], device=device, dtype=torch.float32)
    valid = torch.tensor([True, True, True], device=device, dtype=torch.bool)
    kernel = (3, 3, 3)

    with torch.inference_mode():
        actual_v, actual_k = joint_na3d(q, k, v, kq, kk, kv, times, valid, kernel)
        ref_v, ref_k = _brute_force_joint_na3d(q, k, v, kq, kk, kv, times, valid, kernel)

        video_error = float((actual_v - ref_v).abs().max().item())
        keyframe_error = float((actual_k - ref_k).abs().max().item())
        shape_error = 0.0 if (actual_v.shape == q.shape and actual_k.shape == kq.shape) else 1.0
        finite_error = 0.0 if (torch.isfinite(actual_v).all() and torch.isfinite(actual_k).all()) else 1.0

        invalid = torch.tensor([True, False, True], device=device, dtype=torch.bool)
        inv_v, inv_k = joint_na3d(q, k, v, kq, kk, kv, times, invalid, kernel)
        inv_ref_v, inv_ref_k = _brute_force_joint_na3d(q, k, v, kq, kk, kv, times, invalid, kernel)
        invalid_video_error = float((inv_v - inv_ref_v).abs().max().item())
        invalid_keyframe_error = float((inv_k - inv_ref_k).abs().max().item())
        invalid_plane_zero_error = float(inv_k[:, 1].abs().max().item())

        kv_changed = kv.clone()
        kv_changed[:, 0] = kv_changed[:, 0] + 3.0
        changed_v, _ = joint_na3d(q, k, v, kq, kk, kv_changed, times, valid, kernel)
        influence = float((changed_v - actual_v).abs().max().item())
        influence_error = 0.0 if influence > 1e-6 else 1.0

        tie_query = torch.tensor([1.0], device=device)
        tie_candidates = torch.tensor([0.0, 2.0, 4.0], device=device)
        tie_slots = _nearest_slots(tie_query, tie_candidates, None, 2)
        slot_tie_error = 0.0 if tie_slots.detach().cpu().tolist() == [[0, 1]] else 1.0

        # Keyframe visibility deliberately ignores Kt. At frame 0, the two nearest
        # planes are selected even though plane time 5 is outside a Kt=3 temporal window.
        far_times = torch.tensor([0.0, 5.0], device=device)
        far_valid = torch.tensor([True, True], device=device)
        far_slots = video_keyframe_slots(far_times, far_valid, 1, 2)
        far_visibility_error = 0.0 if far_slots.detach().cpu().tolist() == [[0, 1]] else 1.0

    tolerance = 2e-5 if device.type == "cpu" else 8e-5
    passed = (
        video_error <= tolerance
        and keyframe_error <= tolerance
        and invalid_video_error <= tolerance
        and invalid_keyframe_error <= tolerance
        and invalid_plane_zero_error == 0.0
        and shape_error == 0.0
        and finite_error == 0.0
        and influence_error == 0.0
        and slot_tie_error == 0.0
        and far_visibility_error == 0.0
    )
    return {
        "passed": bool(passed),
        "device": device.type,
        "runtime_backend": selected_joint_attention_backend(q),
        "tolerance": float(tolerance),
        "video_error": video_error,
        "keyframe_error": keyframe_error,
        "invalid_video_error": invalid_video_error,
        "invalid_keyframe_error": invalid_keyframe_error,
        "invalid_plane_zero_error": invalid_plane_zero_error,
        "shape_error": shape_error,
        "finite_error": finite_error,
        "influence": influence,
        "influence_error": influence_error,
        "slot_tie_error": slot_tie_error,
        "far_visibility_error": far_visibility_error,
        "video_shape": tuple(int(x) for x in actual_v.shape),
        "keyframe_shape": tuple(int(x) for x in actual_k.shape),
    }
