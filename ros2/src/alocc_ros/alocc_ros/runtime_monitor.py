"""Measure mock-pipeline ROS latency, rate, and process memory."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import rclpy
from alocc_interfaces.msg import OccupancyFlow3D, OccupancyGrid3D
from rclpy.node import Node

from alocc_ros.config_loader import load_yaml, validate_ros_config
from alocc_ros.metrics import (
    append_csv,
    average,
    fps_from_times,
    percentile,
    read_rss_mb,
    write_summary,
)
from alocc_ros.node_utils import declare_common_parameters
from alocc_ros.prediction_types import measurement_context_for_mode

LATENCY_FIELDS = [
    "kind",
    "timestamp_sec",
    "timestamp_nanosec",
    "receive_time_s",
    "e2e_latency_s",
    "is_mock",
    "prediction_source",
    "rss_mb",
    "measurement_context",
]


class RuntimeMonitor(Node):
    def __init__(self):
        super().__init__("runtime_monitor")
        params = declare_common_parameters(self)
        ros_config = load_yaml(params["ros_config"])
        validate_ros_config(ros_config)
        runtime_dir = Path(params["results_dir"]) / "runtime"
        self._measurement_context = measurement_context_for_mode(params["mode"])
        self._note = (
            "ROS pipeline timings for saved occupancy predictions. "
            "These are not ALOcc inference latency or FPS."
            if params["mode"] == "pkl"
            else "Mock pipeline timings only. These are not ALOcc inference latency or FPS."
        )
        self._latency_path = runtime_dir / "latency.csv"
        self._summary_path = runtime_dir / "summary.csv"
        self._stage_path = runtime_dir / "stage_timings.csv"
        self._receive_times: dict[str, list[float]] = {"occupancy": [], "flow": []}
        self._latencies: dict[str, list[float]] = {"occupancy": [], "flow": []}
        self.create_subscription(
            OccupancyGrid3D, ros_config["occupancy_topic"], self._on_occupancy, 10
        )
        self.create_subscription(
            OccupancyFlow3D, ros_config["flow_topic"], self._on_flow, 10
        )
        self.get_logger().info(self._note)

    def _on_occupancy(self, message: OccupancyGrid3D) -> None:
        self._record("occupancy", message)

    def _on_flow(self, message: OccupancyFlow3D) -> None:
        self._record("flow", message)

    def _record(self, kind: str, message) -> None:
        received = time.time()
        if not bool(message.timestamp_available):
            latency = received - float(message.pipeline_publish_time_s)
        else:
            stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) / 1e9
            latency = received - stamp
        self._receive_times[kind].append(received)
        self._latencies[kind].append(latency)
        append_csv(
            self._latency_path,
            LATENCY_FIELDS,
            {
                "kind": kind,
                "timestamp_sec": int(message.header.stamp.sec),
                "timestamp_nanosec": int(message.header.stamp.nanosec),
                "receive_time_s": f"{received:.9f}",
                "e2e_latency_s": f"{latency:.9f}",
                "is_mock": str(bool(message.is_mock)).lower(),
                "prediction_source": message.prediction_source,
                "rss_mb": f"{read_rss_mb():.6f}",
                "measurement_context": self._measurement_context,
            },
        )
        self._write_summary()

    def _write_summary(self) -> None:
        occupancy = self._latencies["occupancy"]
        flow = self._latencies["flow"]
        stage = _read_stage_columns(self._stage_path)
        metrics = {
            "sample_count": float(len(occupancy) + len(flow)),
            "average_occupancy_e2e_latency_s": average(occupancy),
            "median_occupancy_e2e_latency_s": percentile(occupancy, 50),
            "p95_occupancy_e2e_latency_s": percentile(occupancy, 95),
            "average_flow_e2e_latency_s": average(flow),
            "median_flow_e2e_latency_s": percentile(flow, 50),
            "p95_flow_e2e_latency_s": percentile(flow, 95),
            "occupancy_fps": fps_from_times(self._receive_times["occupancy"]),
            "flow_fps": fps_from_times(self._receive_times["flow"]),
            "average_load_s": average(stage.get("load_s", [])),
            "median_load_s": percentile(stage.get("load_s", []), 50),
            "p95_load_s": percentile(stage.get("load_s", []), 95),
            "average_convert_s": average(stage.get("convert_s", [])),
            "median_convert_s": percentile(stage.get("convert_s", []), 50),
            "p95_convert_s": percentile(stage.get("convert_s", []), 95),
            "average_publish_occupancy_s": average(stage.get("publish_occupancy_s", [])),
            "average_publish_flow_s": average(stage.get("publish_flow_s", [])),
            "rss_mb": read_rss_mb(),
        }
        write_summary(
            self._summary_path,
            metrics,
            self._note,
            measurement_context=self._measurement_context,
        )


def _read_stage_columns(path: Path) -> dict[str, list[float]]:
    columns = {
        "load_s": [],
        "convert_s": [],
        "publish_occupancy_s": [],
        "publish_flow_s": [],
    }
    if not path.is_file():
        return columns
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key in columns:
                if row.get(key):
                    columns[key].append(float(row[key]))
    return columns


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RuntimeMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
