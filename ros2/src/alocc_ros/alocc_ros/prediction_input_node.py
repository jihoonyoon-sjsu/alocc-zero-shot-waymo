"""Load a prediction pair, validate it, and publish occupancy and flow."""

from __future__ import annotations

import time
from pathlib import Path

import rclpy
from alocc_interfaces.msg import OccupancyFlow3D, OccupancyGrid3D
from rclpy.node import Node

from alocc_ros.config_loader import load_yaml, validate_mode, validate_ros_config
from alocc_ros.connectors import create_connector
from alocc_ros.connectors.base import ConnectorUnavailable
from alocc_ros.message_conversion import flow_to_msg, occupancy_to_msg
from alocc_ros.metrics import append_csv
from alocc_ros.node_utils import declare_common_parameters
from alocc_ros.prediction_types import PredictionValidationError, measurement_context_for_mode

STAGE_FIELDS = [
    "timestamp_sec",
    "timestamp_nanosec",
    "sample_index",
    "is_mock",
    "prediction_source",
    "flow_supplied",
    "measurement_context",
    "load_s",
    "convert_s",
    "publish_occupancy_s",
    "publish_flow_s",
]
PUBLISHED_FIELDS = [
    "sample_index",
    "dim_x",
    "dim_y",
    "dim_z",
    "dtype",
    "label_count",
    "label_sha256",
    "flow_supplied",
    "is_mock",
    "prediction_source",
    "sample_metadata",
    "measurement_context",
]
SKIP_FIELDS = ["sample_index", "reason", "measurement_context"]
FLOW_STATUS_FIELDS = ["flow_supplied", "detail", "measurement_context"]


class PredictionInputNode(Node):
    def __init__(self):
        super().__init__("prediction_input_node")
        params = declare_common_parameters(self)
        self._model_config = load_yaml(params["model_config"])
        self._ros_config = load_yaml(params["ros_config"])
        validate_ros_config(self._ros_config)
        validate_mode(params["mode"], self._model_config)
        self._mode = params["mode"]
        self._measurement_context = measurement_context_for_mode(self._mode)
        self._frame_id = self._ros_config["frame_id"]
        runtime_dir = Path(params["results_dir"]) / "runtime"
        self._stage_path = runtime_dir / "stage_timings.csv"
        self._published_path = runtime_dir / "published_occupancy.csv"
        self._skip_path = runtime_dir / "skipped_samples.csv"
        self._flow_status_path = runtime_dir / "flow_status.csv"
        self._flow_absence_reported = False
        self._occupancy_pub = self.create_publisher(
            OccupancyGrid3D, self._ros_config["occupancy_topic"], 10
        )
        self._flow_pub = self.create_publisher(
            OccupancyFlow3D, self._ros_config["flow_topic"], 10
        )
        self._connector = create_connector(
            self._mode, self._model_config, self._ros_config, self
        )
        period = 1.0 / float(self._ros_config["publish_rate_hz"])
        self.create_timer(period, self._tick)
        self.get_logger().info(
            "ALOcc prediction input is running in "
            f"{self._mode} mode. Measurements use context {self._measurement_context}."
        )

    def _tick(self) -> None:
        started_wall = time.time()
        started = time.perf_counter()
        try:
            occupancy, flow = self._connector.load()
        except (ConnectorUnavailable, PredictionValidationError) as exc:
            self.get_logger().warning(str(exc))
            self._record_skips()
            return
        self._record_skips()
        loaded = time.perf_counter()
        try:
            occupancy.validate()
            if flow is not None:
                flow.validate()
        except PredictionValidationError as exc:
            self.get_logger().warning(str(exc))
            return
        occupancy_msg = occupancy_to_msg(occupancy, self._frame_id)
        occupancy_msg.pipeline_publish_time_s = started_wall
        flow_msg = None
        if flow is not None:
            flow_msg = flow_to_msg(flow, self._frame_id)
            flow_msg.pipeline_publish_time_s = started_wall
            flow_msg.sample_index = occupancy.sample_index
            flow_msg.sample_metadata = occupancy.sample_metadata
            flow_msg.timestamp_available = occupancy.timestamp_available
        elif not self._flow_absence_reported:
            self._flow_absence_reported = True
            self.get_logger().warning(
                "Flow was not supplied. Publishing occupancy only and not writing a flow prediction."
            )
            append_csv(
                self._flow_status_path,
                FLOW_STATUS_FIELDS,
                {
                    "flow_supplied": "false",
                    "detail": "flow was not supplied",
                    "measurement_context": self._measurement_context,
                },
            )
        converted = time.perf_counter()
        self._occupancy_pub.publish(occupancy_msg)
        occupancy_published = time.perf_counter()
        if flow_msg is not None:
            self._flow_pub.publish(flow_msg)
        flow_published = time.perf_counter()
        append_csv(
            self._published_path,
            PUBLISHED_FIELDS,
            {
                "sample_index": occupancy.sample_index,
                "dim_x": occupancy.spatial.dim_x,
                "dim_y": occupancy.spatial.dim_y,
                "dim_z": occupancy.spatial.dim_z,
                "dtype": "uint8",
                "label_count": len(occupancy.semantic_labels),
                "label_sha256": _sha256(occupancy.semantic_labels),
                "flow_supplied": str(flow is not None).lower(),
                "is_mock": str(occupancy.is_mock).lower(),
                "prediction_source": occupancy.source,
                "sample_metadata": occupancy.sample_metadata,
                "measurement_context": self._measurement_context,
            },
        )
        append_csv(
            self._stage_path,
            STAGE_FIELDS,
            {
                "timestamp_sec": occupancy.timestamp_sec,
                "timestamp_nanosec": occupancy.timestamp_nanosec,
                "sample_index": occupancy.sample_index,
                "is_mock": str(occupancy.is_mock).lower(),
                "prediction_source": occupancy.source,
                "flow_supplied": str(flow is not None).lower(),
                "measurement_context": self._measurement_context,
                "load_s": f"{loaded - started:.9f}",
                "convert_s": f"{converted - loaded:.9f}",
                "publish_occupancy_s": f"{occupancy_published - converted:.9f}",
                "publish_flow_s": ""
                if flow_msg is None
                else f"{flow_published - occupancy_published:.9f}",
            },
        )

    def _record_skips(self) -> None:
        drain = getattr(self._connector, "drain_skip_records", None)
        if drain is None:
            return
        for record in drain():
            append_csv(
                self._skip_path,
                SKIP_FIELDS,
                {
                    "sample_index": record["sample_index"],
                    "reason": record["reason"],
                    "measurement_context": self._measurement_context,
                },
            )
            self.get_logger().warning(
                f"Skipped pickle sample {record['sample_index']}: {record['reason']}"
            )


def _sha256(labels) -> str:
    import hashlib

    return hashlib.sha256(bytes(labels)).hexdigest()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PredictionInputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
