"""GPU VRAM probing via nvidia-smi.

Works inside the container at the *device* level (per-process memory is hidden
by the container PID namespace, so we read the GPU's total 'used' instead). With
the GPU pinned via compose `device_ids`, the container sees one GPU at index 0
and device-level 'used' ≈ this app's usage.
"""

from __future__ import annotations

import shutil
import subprocess
import threading


def _query(field: str, index: int) -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        return float(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def gpu_mem_used_mib(index: int = 0) -> float | None:
    return _query("memory.used", index)


def gpu_mem_total_mib(index: int = 0) -> float | None:
    return _query("memory.total", index)


class VramSampler(threading.Thread):
    """Background thread that samples device VRAM 'used' to capture a peak."""

    def __init__(self, index: int = 0, interval: float = 0.15):
        super().__init__(daemon=True)
        self.index = index
        self.interval = interval
        self.samples: list[float] = []
        self._stop_evt = threading.Event()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            v = gpu_mem_used_mib(self.index)
            if v is not None:
                self.samples.append(v)
            self._stop_evt.wait(self.interval)

    def stop(self) -> None:
        self._stop_evt.set()

    @property
    def peak(self) -> float | None:
        return max(self.samples) if self.samples else None
