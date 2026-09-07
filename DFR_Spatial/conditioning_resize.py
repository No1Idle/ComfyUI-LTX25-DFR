
"""Strict geometric image preparation utilities for official LTX DFR conditioning."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def bilinear_fill_resize_center_crop(image: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    """Match the official resize-and-center-crop geometry used by LTX DFR conditioning.

    Input IMAGE is Comfy format [B, H, W, C]. Output preserves that format.
    Resize is aspect-preserving "fill" using bilinear interpolation with align_corners=False,
    resized spatial dimensions use ceil(), and the final crop is centered.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError('image must be a torch.Tensor in Comfy IMAGE format [B,H,W,C].')
    if image.ndim != 4:
        raise ValueError(f'Expected IMAGE tensor with 4 dims [B,H,W,C], got shape {tuple(image.shape)}.')
    batch, src_h, src_w, channels = image.shape
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f'Invalid source spatial size {(src_h, src_w)}.')
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f'Invalid target spatial size {(target_height, target_width)}.')

    scale = max(float(target_height) / float(src_h), float(target_width) / float(src_w))
    resized_h = int(math.ceil(float(src_h) * scale))
    resized_w = int(math.ceil(float(src_w) * scale))

    chw = image.permute(0, 3, 1, 2)
    resized = F.interpolate(
        chw,
        size=(resized_h, resized_w),
        mode='bilinear',
        align_corners=False,
    )

    top = max((resized_h - target_height) // 2, 0)
    left = max((resized_w - target_width) // 2, 0)
    bottom = top + target_height
    right = left + target_width
    cropped = resized[:, :, top:bottom, left:right]

    if cropped.shape[2] != target_height or cropped.shape[3] != target_width:
        raise RuntimeError(
            'Centered crop produced unexpected shape '
            f'{tuple(cropped.shape)} for requested {(target_height, target_width)}.'
        )

    return cropped.permute(0, 2, 3, 1).contiguous()
