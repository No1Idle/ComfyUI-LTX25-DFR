"""Small opt-in sampler memory report; never changes the loading policy."""
import logging
import time

import torch
from .operation_profile import ACTIVE_OPERATION_PROFILE

log = logging.getLogger(__name__)


class SamplerMemoryProfile:
    def __init__(self, enabled, device, model, label="Stage 2"):
        self.enabled = bool(enabled) and torch.cuda.is_available() and torch.device(device).type == "cuda"
        self.device = torch.device(device)
        self.model = model
        self.label = label

    def snapshot(self, label):
        if not self.enabled:
            return
        mib = 1024 ** 2
        free, total = torch.cuda.mem_get_info(self.device)
        loaded = getattr(self.model, "loaded_size", None)
        resident = f"{loaded() / mib:.0f}" if callable(loaded) else "unknown"
        log.info(
            "[LTX DFR %s memory] %s: allocated=%.0f MiB; reserved=%.0f MiB; "
            "device_free=%.0f/%.0f MiB; model_loaded=%s MiB",
            self.label, label, torch.cuda.memory_allocated(self.device) / mib,
            torch.cuda.memory_reserved(self.device) / mib, free / mib, total / mib, resident,
        )

    def start_step(self):
        operations = ACTIVE_OPERATION_PROFILE.get()
        if operations is not None:
            operations.reset()
        if self.enabled:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self.started = time.perf_counter()

    def end_step(self, index):
        operations = ACTIVE_OPERATION_PROFILE.get()
        if self.enabled:
            torch.cuda.synchronize(self.device)
            if operations is not None:
                operations.capture(self.device, gap=True)
            log.info(
                "[LTX DFR %s memory] step %d: %.2fs; peak_allocated=%.0f MiB; peak_reserved=%.0f MiB",
                self.label, index + 1, time.perf_counter() - self.started,
                (operations.peak_allocated if operations else torch.cuda.max_memory_allocated(self.device)) / 1024 ** 2,
                (operations.peak_reserved if operations else torch.cuda.max_memory_reserved(self.device)) / 1024 ** 2,
            )
            self.snapshot(f"after step {index + 1}")
        if operations is not None:
            operations.report(index)
