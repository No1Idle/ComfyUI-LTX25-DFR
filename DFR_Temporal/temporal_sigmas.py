"""Sigma schedule for Temporal DFR.

The official temporal rounds use ``DISTILLED_SIGMAS[4:]`` and advance these
points with the official eta=0.5 rectified-flow Euler-Ancestral step.
"""

from __future__ import annotations

import torch

if "." in (__package__ or ""):  # Normal Comfy custom-node package import.
    from ..DFR_Spatial.dfr_sigmas import validate_custom_sigma_schedule
else:  # Standalone temporal regression tests.
    from DFR_Spatial.dfr_sigmas import validate_custom_sigma_schedule


TEMPORAL_DISTILLED_SIGMA_VALUES = (
    0.975,
    0.909375,
    0.725,
    0.421875,
    0.0,
)


def temporal_sigmas() -> torch.Tensor:
    """Return the exact official Temporal-DFR sigma points as float32."""
    return torch.tensor(TEMPORAL_DISTILLED_SIGMA_VALUES, dtype=torch.float32)


def validate_temporal_sigma_schedule(sigmas: torch.Tensor) -> torch.Tensor:
    """Validate an advanced Temporal-DFR schedule override."""
    return validate_custom_sigma_schedule(sigmas, "Custom Temporal-DFR sigmas")
