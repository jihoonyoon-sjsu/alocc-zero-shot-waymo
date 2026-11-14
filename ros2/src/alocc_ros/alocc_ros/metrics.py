"""Runtime statistics for the mock ROS pipeline.

These numbers describe framework overhead with synthetic predictions.
They are not ALOcc model inference performance.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from alocc_ros.prediction_types import MEASUREMENT_CONTEXT


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: list[float], percent: float) -> float | None:
    """Linear interpolation percentile. percent is in [0, 100]."""
    if not values:
        return None
    if percent < 0.0 or percent > 100.0:
        raise ValueError("percent must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percent / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def fps_from_times(times: list[float]) -> float | None:
    if len(times) < 2:
        return None
    span = times[-1] - times[0]
    if span <= 0.0:
        return None
    return (len(times) - 1) / span


def read_rss_mb() -> float:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS was not present in /proc/self/status")


def append_csv(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary(
    path: Path,
    metrics: dict[str, float | None],
    note: str,
    measurement_context: str = MEASUREMENT_CONTEXT,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "value", "unit", "measurement_context", "note"],
        )
        writer.writeheader()
        for metric, value in metrics.items():
            writer.writerow(
                {
                    "metric": metric,
                    "value": "" if value is None else f"{value:.9f}",
                    "unit": _unit_for(metric),
                    "measurement_context": measurement_context,
                    "note": note,
                }
            )


def _unit_for(metric: str) -> str:
    if metric.endswith("_s"):
        return "seconds"
    if metric == "fps" or metric.endswith("_fps"):
        return "frames_per_second"
    if metric.endswith("_mb"):
        return "mebibytes"
    if metric.endswith("_count"):
        return "count"
    return ""
