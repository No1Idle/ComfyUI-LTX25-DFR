"""Exact Lightricks Spatial-DFR distilled sigma schedules.

Reference:
  Lightricks/LTX-2
  packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py

The official DFR pipeline passes these tensors directly into Stage 1 and Stage 2.
There is no LTXVScheduler/token-dependent rescheduling in this parity path.
"""

from __future__ import annotations

import torch


# Keep the decimal values exactly as upstream. torch.tensor() defaults to
# float32 upstream; we state the dtype explicitly so the Comfy output is stable.
DISTILLED_SIGMA_VALUES = (
    1.0,
    0.99375,
    0.9875,
    0.98125,
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)

STAGE_2_DISTILLED_SIGMA_VALUES = (
    0.909375,
    0.725,
    0.421875,
    0.0,
)

# Non-parity experiment: keep every official Stage-2 point and split only the
# largest remaining Euler interval (0.421875 -> 0) at its arithmetic midpoint.
# This preserves the official initial noise level and the first two distilled
# transitions, so the comparison changes exactly one thing: one extra late
# Stage-2 transformer evaluation / Euler transition.
EXPERIMENTAL_STAGE_2_EXTRA_STEP_SIGMA_VALUES = (
    0.909375,
    0.725,
    0.421875,
    0.2109375,
    0.0,
)


def stage_1_sigmas() -> torch.Tensor:
    """Return a fresh float32 tensor matching upstream DISTILLED_SIGMAS."""
    return torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32)


def stage_2_sigmas() -> torch.Tensor:
    """Return a fresh float32 tensor matching upstream STAGE_2_DISTILLED_SIGMAS."""
    return torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32)


def experimental_stage_2_extra_step_sigmas() -> torch.Tensor:
    """Return the fixed 5-point / 4-transition Stage-2 comparison schedule."""
    return torch.tensor(EXPERIMENTAL_STAGE_2_EXTRA_STEP_SIGMA_VALUES, dtype=torch.float32)


def validate_custom_sigma_schedule(sigmas: torch.Tensor, name: str) -> torch.Tensor:
    """Validate an advanced DFR schedule override and normalize it to float32."""
    if not torch.is_tensor(sigmas):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(sigmas).__name__}.")
    if sigmas.ndim != 1:
        raise ValueError(f"{name} must be rank 1, got shape={tuple(sigmas.shape)}.")
    if sigmas.numel() < 2:
        raise ValueError(f"{name} must contain at least 2 entries, got {sigmas.numel()}.")
    schedule = sigmas.detach().to(dtype=torch.float32)
    if not torch.isfinite(schedule).all().item():
        raise ValueError(f"{name} must contain only finite values.")
    if torch.any(schedule < 0).item():
        raise ValueError(f"{name} must be non-negative.")
    if float(schedule[-1].item()) != 0.0:
        raise ValueError(f"{name} must end at 0.0.")
    if torch.any(schedule[:-1] <= 0).item():
        raise ValueError(f"Every non-terminal value in {name} must be greater than zero.")
    if torch.any(schedule[1:] > schedule[:-1]).item():
        raise ValueError(f"{name} must be monotonically non-increasing.")
    return schedule


def validate_sigma_tensor(actual: torch.Tensor, expected: torch.Tensor, name: str) -> tuple[bool, float, str]:
    if not torch.is_tensor(actual):
        return False, float("inf"), f"{name}: expected torch.Tensor, got {type(actual).__name__}"
    if actual.ndim != 1:
        return False, float("inf"), f"{name}: expected rank-1 tensor, got shape={tuple(actual.shape)}"
    if tuple(actual.shape) != tuple(expected.shape):
        return False, float("inf"), f"{name}: shape={tuple(actual.shape)}, expected={tuple(expected.shape)}"

    actual_f32 = actual.detach().to(device="cpu", dtype=torch.float32)
    expected_f32 = expected.detach().to(device="cpu", dtype=torch.float32)
    max_error = float((actual_f32 - expected_f32).abs().max().item()) if actual_f32.numel() else 0.0
    exact = torch.equal(actual_f32, expected_f32)
    dtype_ok = actual.dtype == torch.float32
    endpoints_ok = actual_f32[0].item() > 0.0 and actual_f32[-1].item() == 0.0
    monotonic_ok = bool(torch.all(actual_f32[:-1] >= actual_f32[1:]).item())
    passed = exact and dtype_ok and endpoints_ok and monotonic_ok
    details = (
        f"{name}: exact={exact}; dtype={actual.dtype}; dtype_float32={dtype_ok}; "
        f"count={actual.numel()}; monotonic_nonincreasing={monotonic_ok}; "
        f"first={float(actual_f32[0]):.9g}; last={float(actual_f32[-1]):.9g}; "
        f"max_abs_error={max_error:.9g}"
    )
    return passed, max_error, details
