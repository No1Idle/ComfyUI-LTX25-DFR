"""Temporary low-overhead runtime profiling for the consolidated Stage-1 path."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import time
from typing import Iterator

import torch


@dataclass
class _ProfileRecord:
    name: str
    cpu_seconds: float
    cuda_start: torch.cuda.Event | None
    cuda_end: torch.cuda.Event | None


class Stage1RuntimeProfiler:
    """Collect section timings without synchronizing between denoising steps."""

    def __init__(self, *, enabled: bool, device: torch.device):
        self.enabled = bool(enabled)
        self.device = torch.device(device)
        self._cuda_enabled = self.enabled and self.device.type == "cuda" and torch.cuda.is_available()
        self._records: list[_ProfileRecord] = []
        self._started = time.perf_counter()
        self._finished: float | None = None

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            with nullcontext():
                yield
            return

        start_event = end_event = None
        if self._cuda_enabled:
            with torch.cuda.device(self.device):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
        cpu_started = time.perf_counter()
        try:
            yield
        finally:
            cpu_seconds = time.perf_counter() - cpu_started
            if end_event is not None:
                with torch.cuda.device(self.device):
                    end_event.record()
            self._records.append(
                _ProfileRecord(
                    name=str(name),
                    cpu_seconds=cpu_seconds,
                    cuda_start=start_event,
                    cuda_end=end_event,
                )
            )

    def format_report(self) -> str:
        if not self.enabled:
            return "Stage-1 profiling is disabled."
        if self._finished is None:
            if self._cuda_enabled:
                torch.cuda.synchronize(self.device)
            self._finished = time.perf_counter()

        cpu_totals: dict[str, float] = defaultdict(float)
        cuda_totals_ms: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for record in self._records:
            counts[record.name] += 1
            cpu_totals[record.name] += record.cpu_seconds
            if record.cuda_start is not None and record.cuda_end is not None:
                cuda_totals_ms[record.name] += float(record.cuda_start.elapsed_time(record.cuda_end))

        order: list[str] = []
        for record in self._records:
            if record.name not in order:
                order.append(record.name)

        lines = [
            "LTX Spatial DFR Stage 1 profile",
            f"  device: {self.device}",
            f"  total synchronized wall: {self._finished - self._started:.3f} s",
        ]
        for name in order:
            count_suffix = f" x{counts[name]}" if counts[name] > 1 else ""
            if self._cuda_enabled:
                lines.append(
                    f"  {name}{count_suffix}: host={cpu_totals[name]:.3f} s; "
                    f"cuda={cuda_totals_ms[name] / 1000.0:.3f} s"
                )
            else:
                lines.append(f"  {name}{count_suffix}: wall={cpu_totals[name]:.3f} s")

        transformer_records = [record for record in self._records if record.name == "transformer forward"]
        if len(transformer_records) > 1:
            timing_kind = "cuda" if self._cuda_enabled else "wall"
            lines.append(f"  transformer forward per step ({timing_kind}):")
            for step_index, record in enumerate(transformer_records, start=1):
                if self._cuda_enabled and record.cuda_start is not None and record.cuda_end is not None:
                    seconds = float(record.cuda_start.elapsed_time(record.cuda_end)) / 1000.0
                else:
                    seconds = record.cpu_seconds
                lines.append(f"    step {step_index}: {seconds:.3f} s")
        if self._cuda_enabled:
            lines.append(
                "  note: host values measure Python/dispatch time; CUDA values measure queued GPU work. "
                "Model loading/offload can include host work not represented by CUDA events."
            )
        return "\n".join(lines)


def profile_section(profiler: Stage1RuntimeProfiler | None, name: str):
    return profiler.section(name) if profiler is not None else nullcontext()
