import torch
import comfy.utils


def _get_noise_mask(latent):
    """Clone an existing ComfyUI noise mask, or create an all-1 mask."""
    samples = latent["samples"]
    noise_mask = latent.get("noise_mask", None)
    if noise_mask is None:
        return torch.ones(
            (samples.shape[0], 1, samples.shape[2], 1, 1),
            dtype=torch.float32,
            device=samples.device,
        )
    return noise_mask.clone()


def _resolve_pixel_frame_to_latent_index(frame_idx, latent_length, time_scale_factor):
    """
    LTX temporal layout is causal/asymmetric:
      pixel frame 0 -> latent time index 0
      following frames are temporally compressed by time_scale_factor (normally 8).

    Negative frame indices are resolved from the end, so -1 means the last output frame.
    """
    pixel_length = (latent_length - 1) * time_scale_factor + 1

    if frame_idx < 0:
        resolved_frame_idx = pixel_length + frame_idx
    else:
        resolved_frame_idx = frame_idx

    if resolved_frame_idx < 0 or resolved_frame_idx >= pixel_length:
        raise ValueError(
            f"frame_idx {frame_idx} resolves to pixel frame {resolved_frame_idx}, "
            f"but this latent represents {pixel_length} output frames (0..{pixel_length - 1})."
        )

    latent_idx = (resolved_frame_idx + time_scale_factor - 1) // time_scale_factor
    return resolved_frame_idx, latent_idx, pixel_length


def _encode_single_image_for_latent(vae, image, samples):
    if image.shape[0] != 1:
        raise ValueError(
            "This node intentionally accepts exactly one IMAGE at a time. "
            "Use separate chained nodes for separate keyframes."
        )

    scale_factors = vae.downscale_index_formula
    time_scale_factor, height_scale_factor, width_scale_factor = scale_factors

    _, _, _, latent_height, latent_width = samples.shape
    width = latent_width * width_scale_factor
    height = latent_height * height_scale_factor

    if image.shape[1] != height or image.shape[2] != width:
        pixels = comfy.utils.common_upscale(
            image.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
    else:
        pixels = image

    encoded = vae.encode(pixels[:, :, :, :3])

    if encoded.ndim != 5:
        raise ValueError(f"Unexpected VAE latent shape: {tuple(encoded.shape)}")
    if encoded.shape[2] != 1:
        raise ValueError(
            f"Expected a single-image encode to produce one latent time slice, got {encoded.shape[2]}."
        )
    if encoded.shape[1] != samples.shape[1]:
        raise ValueError(
            f"VAE produced {encoded.shape[1]} channels but target latent has {samples.shape[1]} channels."
        )
    if encoded.shape[-2:] != samples.shape[-2:]:
        raise ValueError(
            f"Encoded spatial latent size {tuple(encoded.shape[-2:])} does not match target "
            f"{tuple(samples.shape[-2:])}."
        )

    if encoded.shape[0] == 1 and samples.shape[0] > 1:
        encoded = encoded.expand(samples.shape[0], -1, -1, -1, -1)
    elif encoded.shape[0] != samples.shape[0]:
        raise ValueError(
            f"Image latent batch {encoded.shape[0]} does not match video latent batch {samples.shape[0]}."
        )

    # Be explicit for blending/assignment on systems where the VAE and latent may
    # temporarily live on different devices/dtypes because of offloading.
    encoded = encoded.to(device=samples.device, dtype=samples.dtype)

    return encoded, int(time_scale_factor)


class CropImageToMaskBBox:
    """Crop an IMAGE to the thresholded MASK bounding box, with pixel expansion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "expand_px": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Expand the detected mask bounding box by this many pixels on every side. Expansion is clamped to the image boundaries.",
                    },
                ),
                "alpha_threshold": (
                    "FLOAT",
                    {
                        "default": 0.01,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Only mask pixels strictly above this value are used to compute the bounding box. 0.0 includes every non-zero mask pixel.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("cropped_image", "x", "y", "width", "height")
    FUNCTION = "crop"
    CATEGORY = "image/crop"
    DESCRIPTION = (
        "Find the smallest rectangular bounding box containing mask pixels above alpha_threshold, "
        "expand it by expand_px in all directions while respecting image boundaries, and crop the image. "
        "Also returns x, y, width, and height matching the crop rectangle."
    )

    def crop(self, image, mask, expand_px=0, alpha_threshold=0.01):
        if not torch.is_tensor(image) or image.ndim != 4:
            raise ValueError(
                f"Expected IMAGE shaped [B,H,W,C], got {getattr(image, 'shape', None)}."
            )
        if not torch.is_tensor(mask):
            raise ValueError("Expected MASK to be a torch tensor.")

        # Normal ComfyUI MASK is [B,H,W]. Also accept [H,W] and the occasional
        # singleton-channel variants [B,1,H,W] / [B,H,W,1].
        if mask.ndim == 2:
            mask_2d_batch = mask.unsqueeze(0)
        elif mask.ndim == 3:
            mask_2d_batch = mask
        elif mask.ndim == 4 and mask.shape[1] == 1:
            mask_2d_batch = mask[:, 0, :, :]
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask_2d_batch = mask[:, :, :, 0]
        else:
            raise ValueError(
                f"Expected MASK shaped [H,W], [B,H,W], [B,1,H,W], or [B,H,W,1], got {tuple(mask.shape)}."
            )

        image_height = int(image.shape[1])
        image_width = int(image.shape[2])
        mask_height = int(mask_2d_batch.shape[-2])
        mask_width = int(mask_2d_batch.shape[-1])

        if mask_height != image_height or mask_width != image_width:
            raise ValueError(
                f"MASK size {mask_width}x{mask_height} does not match IMAGE size "
                f"{image_width}x{image_height}."
            )

        alpha_threshold = float(alpha_threshold)
        expand_px = int(expand_px)

        active = mask_2d_batch > alpha_threshold
        coords = torch.nonzero(active, as_tuple=False)
        if coords.numel() == 0:
            raise ValueError(
                f"MASK has no pixels above alpha_threshold={alpha_threshold}. "
                "Lower the threshold or provide a non-empty mask."
            )

        # coords columns for [B,H,W] are batch, y, x. Taking the extrema over all
        # mask batches gives one crop rectangle, which is required because an IMAGE
        # batch must have a common HxW tensor shape.
        y_min = int(coords[:, -2].min().item())
        y_max = int(coords[:, -2].max().item())
        x_min = int(coords[:, -1].min().item())
        x_max = int(coords[:, -1].max().item())

        x = max(0, x_min - expand_px)
        y = max(0, y_min - expand_px)
        right = min(image_width, x_max + 1 + expand_px)
        bottom = min(image_height, y_max + 1 + expand_px)

        width = right - x
        height = bottom - y
        cropped = image[:, y:bottom, x:right, :]

        return (cropped, x, y, width, height)



def _parse_float_list(value, field_name):
    """Parse comma/semicolon/whitespace-separated numeric values from a string."""
    if isinstance(value, (int, float)):
        return [float(value)]

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string of numbers, got {type(value).__name__}.")

    cleaned = value.replace(",", " ").replace(";", " ")
    parts = [part for part in cleaned.split() if part]
    if not parts:
        raise ValueError(f"{field_name} is empty. Enter one or more numeric values.")

    values = []
    for part in parts:
        try:
            values.append(float(part))
        except ValueError as exc:
            raise ValueError(
                f"Invalid number '{part}' in {field_name}. "
                "Use comma, semicolon, spaces, or new lines between values."
            ) from exc
    return values


def _expand_single_or_match(values, count, field_name):
    """A single value applies to all images; otherwise the count must match exactly."""
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(
            f"{field_name} must contain either 1 value or exactly {count} values "
            f"(one per image); got {len(values)}."
        )
    return values


def _split_image_batch(images):
    """Normalize a normal ComfyUI IMAGE batch, or a list/tuple of IMAGE tensors, to single images."""
    items = []

    def add_tensor(tensor):
        if not torch.is_tensor(tensor):
            raise ValueError(
                f"images must contain IMAGE tensors, got {type(tensor).__name__}."
            )
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(
                f"Expected IMAGE shaped [B,H,W,C] (or one [H,W,C] image), got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] < 1:
            raise ValueError("images contains an empty IMAGE batch.")
        for i in range(tensor.shape[0]):
            items.append(tensor[i : i + 1])

    if torch.is_tensor(images):
        add_tensor(images)
    elif isinstance(images, (list, tuple)):
        for item in images:
            add_tensor(item)
    else:
        raise ValueError(
            f"images must be an IMAGE batch or a list/tuple of IMAGE tensors, got {type(images).__name__}."
        )

    if not items:
        raise ValueError("No images were provided.")
    return items



NODE_CLASS_MAPPINGS = {
    "CropImageToMaskBBox_Current": CropImageToMaskBBox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CropImageToMaskBBox_Current": "Crop Image to Mask Bounding Box",
}

# -----------------------------------------------------------------------------
# Spatial DFR parity path (strict Phase A)
# Kept in a dedicated subdirectory so the existing experimental Blend/Replace
# nodes remain isolated from the official Lightricks-conditioning port.
# -----------------------------------------------------------------------------
from .DFR_Spatial.nodes import (
    NODE_CLASS_MAPPINGS as DFR_SPATIAL_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as DFR_SPATIAL_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(DFR_SPATIAL_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DFR_SPATIAL_NODE_DISPLAY_NAME_MAPPINGS)

# -----------------------------------------------------------------------------
# Temporal DFR path
# Temporal-only implementation stays isolated in DFR_Temporal and consumes the
# finalized Spatial-DFR handoffs as upstream boundaries.
# -----------------------------------------------------------------------------
from .DFR_Temporal.nodes import (
    NODE_CLASS_MAPPINGS as DFR_TEMPORAL_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as DFR_TEMPORAL_NODE_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(DFR_TEMPORAL_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DFR_TEMPORAL_NODE_DISPLAY_NAME_MAPPINGS)
