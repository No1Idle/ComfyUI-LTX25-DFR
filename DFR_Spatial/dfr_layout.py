"""Exact Spatial DFR canvas-layout helpers ported from Lightricks/LTX-2.

Reference:
  packages/ltx-pipelines/src/ltx_pipelines/dfr_layout.py

Only the spatial/base-canvas helpers are included at this stage. TemporalTilePlan
belongs to the later Temporal DFR phase.
"""

from __future__ import annotations

from typing import Any


SEGMENT_CANDIDATES = (24, 32)
DEFAULT_TEMPORAL_SCALE = 8
DFR_LAYOUT_VERSION = 1
EXPERIMENTAL_DENSE_SLOT_LAYOUT_VERSION = 2
EXPERIMENTAL_DENSE_SLOT_LAYOUT_KIND = "experimental_dense_generated_slots_16"
EXPERIMENTAL_DENSE_SLOT_SEGMENT = 16
CUSTOM_GENERATED_SLOT_LAYOUT_VERSION = 3
CUSTOM_GENERATED_SLOT_LAYOUT_KIND = "custom_generated_slots"
CONTINUATION_LAYOUT_VERSION = 4
CONTINUATION_LAYOUT_KIND = "continuation_canvas"


def padding_to_segment(content_frames: int, segment: int) -> int:
    """Frames needed to round content_frames up to a whole segment."""
    return (-content_frames) % segment


def choose_segment_length(content_frames: int) -> int:
    """Exact Lightricks selection: least padding; ties prefer the larger segment."""
    if content_frames < 1:
        raise ValueError(f"content_frames must be >= 1, got {content_frames}")
    return min(
        SEGMENT_CANDIDATES,
        key=lambda segment: (padding_to_segment(content_frames, segment), -segment),
    )


def resolve_canvas(
    num_frames: int,
    *,
    temporal_scale: int = DEFAULT_TEMPORAL_SCALE,
) -> tuple[int, int, list[int]]:
    """Exact Lightricks Spatial-DFR canvas resolver.

    Returns ``(padded_num_frames, segment_length, generated_slot_positions)``.
    ``num_frames`` is the caller-requested pixel-frame count. The function pads
    ``num_frames - 1`` to a whole 24/32-frame segment grid. Frame 0 is excluded
    from generated slots; the padded terminal frame is included.
    """
    num_frames = int(num_frames)
    temporal_scale = int(temporal_scale)
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if temporal_scale < 1:
        raise ValueError(f"temporal_scale must be >= 1, got {temporal_scale}")
    if (num_frames - 1) % temporal_scale != 0:
        raise ValueError(
            f"num_frames must satisfy (num_frames - 1) % {temporal_scale} == 0 (got {num_frames})"
        )

    content = num_frames - 1
    if content == 0:
        raise ValueError("The canvas needs at least 2 pixel frames")

    segment = choose_segment_length(content)
    content_padded = content + padding_to_segment(content, segment)
    positions = [segment * index for index in range(1, content_padded // segment + 1)]
    return content_padded + 1, segment, positions


def resolve_experimental_dense_slot_canvas(
    num_frames: int,
    *,
    temporal_scale: int = DEFAULT_TEMPORAL_SCALE,
) -> tuple[int, int, list[int]]:
    """Resolve an explicitly experimental 16-frame generated-slot canvas.

    This is deliberately separate from :func:`resolve_canvas`, whose official
    24/32 selection remains frozen.  The experimental segment must stay on an
    LTX temporal-latent border so every generated slot has a valid position.
    """
    num_frames = int(num_frames)
    temporal_scale = int(temporal_scale)
    segment = EXPERIMENTAL_DENSE_SLOT_SEGMENT
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if temporal_scale < 1:
        raise ValueError(f"temporal_scale must be >= 1, got {temporal_scale}")
    if segment % temporal_scale != 0:
        raise ValueError(
            f"experimental segment {segment} must be divisible by temporal_scale {temporal_scale}"
        )
    if (num_frames - 1) % temporal_scale != 0:
        raise ValueError(
            f"num_frames must satisfy (num_frames - 1) % {temporal_scale} == 0 (got {num_frames})"
        )

    content = num_frames - 1
    if content == 0:
        raise ValueError("The canvas needs at least 2 pixel frames")
    content_padded = content + padding_to_segment(content, segment)
    positions = [segment * index for index in range(1, content_padded // segment + 1)]
    return content_padded + 1, segment, positions


def pixel_to_latent_index(pixel_frame: int, temporal_scale: int = DEFAULT_TEMPORAL_SCALE) -> int:
    """Exact Lightricks xN-border pixel-frame -> latent-index mapping."""
    pixel_frame = int(pixel_frame)
    temporal_scale = int(temporal_scale)
    if pixel_frame < 0:
        raise ValueError(f"pixel_frame must be >= 0, got {pixel_frame}")
    if pixel_frame != 0 and pixel_frame % temporal_scale != 0:
        raise ValueError(f"pixel_frame {pixel_frame} is not on the x{temporal_scale} latent border")
    return pixel_frame // temporal_scale


def make_layout(num_frames: int, temporal_scale: int = DEFAULT_TEMPORAL_SCALE) -> dict[str, Any]:
    """Create a serializable Comfy-side representation of the exact DFR layout."""
    requested_frames = int(num_frames)
    temporal_scale = int(temporal_scale)
    padded_frames, segment_length, positions = resolve_canvas(
        requested_frames,
        temporal_scale=temporal_scale,
    )
    return {
        "version": DFR_LAYOUT_VERSION,
        "requested_frames": requested_frames,
        "padded_frames": padded_frames,
        "padding_frames": padded_frames - requested_frames,
        "segment_length": segment_length,
        "pixel_frame_indices": tuple(positions),
        "temporal_scale": temporal_scale,
        "requested_latent_frames": (requested_frames - 1) // temporal_scale + 1,
        "padded_latent_frames": (padded_frames - 1) // temporal_scale + 1,
    }


def _validate_custom_generated_slot_indices(
    pixel_frame_indices: list[int] | tuple[int, ...],
    *,
    padded_frames: int,
    temporal_scale: int,
) -> tuple[int, ...]:
    positions = tuple(int(position) for position in pixel_frame_indices)
    if not positions:
        raise ValueError("Custom generated_slot_indices must contain at least one pixel-frame index.")
    if positions[0] <= 0:
        raise ValueError(
            "Custom generated_slot_indices must be greater than zero; frame 0 is the causal base frame."
        )
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError(
            f"Custom generated_slot_indices must be strictly increasing and unique, got {positions}."
        )
    if positions[-1] >= int(padded_frames):
        raise ValueError(
            "Custom generated_slot_indices must lie inside the padded DFR canvas: "
            f"last index {positions[-1]}, padded_frames={int(padded_frames)}."
        )
    off_border = tuple(position for position in positions if position % int(temporal_scale) != 0)
    if off_border:
        raise ValueError(
            "Custom generated_slot_indices must lie on the temporal latent border "
            f"(multiples of {int(temporal_scale)}); invalid indices={off_border}."
        )
    return positions


def make_custom_generated_slot_layout(
    num_frames: int,
    pixel_frame_indices: list[int] | tuple[int, ...],
    temporal_scale: int = DEFAULT_TEMPORAL_SCALE,
) -> dict[str, Any]:
    """Use official canvas geometry with an explicit generated-slot position plan."""
    layout = make_layout(num_frames, temporal_scale)
    positions = _validate_custom_generated_slot_indices(
        pixel_frame_indices,
        padded_frames=int(layout["padded_frames"]),
        temporal_scale=int(layout["temporal_scale"]),
    )
    layout["version"] = CUSTOM_GENERATED_SLOT_LAYOUT_VERSION
    layout["layout_kind"] = CUSTOM_GENERATED_SLOT_LAYOUT_KIND
    layout["pixel_frame_indices"] = positions
    return layout


def make_experimental_dense_slot_layout(
    num_frames: int,
    temporal_scale: int = DEFAULT_TEMPORAL_SCALE,
) -> dict[str, Any]:
    """Create the isolated experimental layout with generated slots every 16 frames."""
    requested_frames = int(num_frames)
    temporal_scale = int(temporal_scale)
    padded_frames, segment_length, positions = resolve_experimental_dense_slot_canvas(
        requested_frames,
        temporal_scale=temporal_scale,
    )
    return {
        "version": EXPERIMENTAL_DENSE_SLOT_LAYOUT_VERSION,
        "layout_kind": EXPERIMENTAL_DENSE_SLOT_LAYOUT_KIND,
        "requested_frames": requested_frames,
        "padded_frames": padded_frames,
        "padding_frames": padded_frames - requested_frames,
        "segment_length": segment_length,
        "pixel_frame_indices": tuple(positions),
        "temporal_scale": temporal_scale,
        "requested_latent_frames": (requested_frames - 1) // temporal_scale + 1,
        "padded_latent_frames": (padded_frames - 1) // temporal_scale + 1,
    }


def make_continuation_layout(requested_frames, padded_frames, pixel_frame_indices, temporal_scale=8):
    """Preserve an existing temporal canvas rather than selecting new padding."""
    requested_frames, padded_frames, temporal_scale = map(int, (requested_frames, padded_frames, temporal_scale))
    if temporal_scale < 1 or not 1 < requested_frames <= padded_frames:
        raise ValueError("Invalid continuation frame counts or temporal scale")
    if (requested_frames-1) % temporal_scale or (padded_frames-1) % temporal_scale:
        raise ValueError("Continuation frame counts must lie on the latent grid")
    positions = _validate_custom_generated_slot_indices(
        pixel_frame_indices, padded_frames=padded_frames, temporal_scale=temporal_scale)
    return dict(version=CONTINUATION_LAYOUT_VERSION, layout_kind=CONTINUATION_LAYOUT_KIND, requested_frames=requested_frames,
                padded_frames=padded_frames, padding_frames=padded_frames-requested_frames,
                segment_length=0, pixel_frame_indices=positions, temporal_scale=temporal_scale,
                requested_latent_frames=(requested_frames-1)//temporal_scale+1,
                padded_latent_frames=(padded_frames-1)//temporal_scale+1)


def validate_layout(layout: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(layout, dict):
        raise ValueError(f"DFR layout must be a dict, got {type(layout).__name__}.")
    version = layout.get("version")
    if version == CONTINUATION_LAYOUT_VERSION:
        if layout.get("layout_kind") != CONTINUATION_LAYOUT_KIND:
            raise ValueError("Invalid continuation layout kind")
        reference = make_continuation_layout(layout["requested_frames"], layout["padded_frames"],
                                             layout["pixel_frame_indices"], layout["temporal_scale"])
    elif version == DFR_LAYOUT_VERSION:
        requested = int(layout["requested_frames"])
        temporal_scale = int(layout["temporal_scale"])
        reference = make_layout(requested, temporal_scale)
    elif version == EXPERIMENTAL_DENSE_SLOT_LAYOUT_VERSION:
        if layout.get("layout_kind") != EXPERIMENTAL_DENSE_SLOT_LAYOUT_KIND:
            raise ValueError(
                "Malformed experimental DFR layout kind: "
                f"got {layout.get('layout_kind')!r}, expected {EXPERIMENTAL_DENSE_SLOT_LAYOUT_KIND!r}."
            )
        requested = int(layout["requested_frames"])
        temporal_scale = int(layout["temporal_scale"])
        reference = make_experimental_dense_slot_layout(requested, temporal_scale)
    elif version == CUSTOM_GENERATED_SLOT_LAYOUT_VERSION:
        if layout.get("layout_kind") != CUSTOM_GENERATED_SLOT_LAYOUT_KIND:
            raise ValueError(
                "Malformed custom DFR layout kind: "
                f"got {layout.get('layout_kind')!r}, expected {CUSTOM_GENERATED_SLOT_LAYOUT_KIND!r}."
            )
        requested = int(layout["requested_frames"])
        temporal_scale = int(layout["temporal_scale"])
        reference = make_custom_generated_slot_layout(
            requested,
            layout.get("pixel_frame_indices", ()),
            temporal_scale,
        )
    else:
        raise ValueError(
            f"Unsupported DFR layout version {version}; expected official version {DFR_LAYOUT_VERSION}, "
            f"experimental version {EXPERIMENTAL_DENSE_SLOT_LAYOUT_VERSION}, "
            f"custom version {CUSTOM_GENERATED_SLOT_LAYOUT_VERSION}, or continuation version 4."
        )
    for key in (
        "padded_frames",
        "padding_frames",
        "segment_length",
        "pixel_frame_indices",
        "requested_latent_frames",
        "padded_latent_frames",
    ):
        if layout.get(key) != reference[key]:
            raise ValueError(
                f"Malformed DFR layout field {key}: got {layout.get(key)!r}, expected {reference[key]!r}."
            )
    return layout


def layout_from_handoff(handoff):
    """Get the authoritative layout carried through a spatial continuation."""
    source = getattr(handoff, "stage_1_handoff", handoff)
    layout = getattr(source, "dfr_layout", None)
    if layout is None:
        raise ValueError("This handoff has no embedded DFR layout. Rerun Stage 1 and the spatial upscaler, or supply the source layout to the modular upscaler.")
    return dict(validate_layout(layout))
