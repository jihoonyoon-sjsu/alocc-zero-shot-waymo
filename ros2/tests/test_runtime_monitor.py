from pathlib import Path

import pytest

from alocc_ros.metrics import (
    append_csv,
    average,
    fps_from_times,
    percentile,
    read_rss_mb,
    write_summary,
)
from alocc_ros.prediction_types import MEASUREMENT_CONTEXT


def test_latency_statistics():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert average(values) == 3.0
    assert percentile(values, 50) == 3.0
    assert percentile(values, 95) == pytest.approx(4.8)
    assert fps_from_times([0.0, 0.5, 1.0]) == 2.0
    assert fps_from_times([1.0]) is None


def test_memory_reading_is_positive():
    assert read_rss_mb() > 0.0


def test_summary_csv_marks_mock_context(tmp_path: Path):
    path = tmp_path / "summary.csv"
    write_summary(
        path,
        {"average_occupancy_e2e_latency_s": 0.01, "rss_mb": 12.5, "occupancy_fps": None},
        "Mock pipeline timings only.",
    )
    text = path.read_text(encoding="utf-8")
    assert MEASUREMENT_CONTEXT in text
    assert "Mock pipeline timings only." in text
    stage = tmp_path / "stage.csv"
    append_csv(stage, ["load_s"], {"load_s": "0.001"})
    assert "0.001" in stage.read_text(encoding="utf-8")

