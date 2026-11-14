"""Validate occupancy and flow messages after they arrive on ROS topics."""

from __future__ import annotations

from pathlib import Path

import rclpy
from alocc_interfaces.msg import OccupancyFlow3D, OccupancyGrid3D
from rclpy.node import Node

from alocc_ros.config_loader import load_yaml, validate_ros_config
from alocc_ros.message_conversion import flow_from_msg, occupancy_from_msg
from alocc_ros.metrics import append_csv
from alocc_ros.node_utils import declare_common_parameters
from alocc_ros.prediction_types import PredictionValidationError, measurement_context_for_mode

FIELDS = [
    "timestamp_sec",
    "timestamp_nanosec",
    "kind",
    "passed",
    "is_mock",
    "prediction_source",
    "detail",
    "measurement_context",
]


class PredictionValidator(Node):
    def __init__(self):
        super().__init__("prediction_validator")
        params = declare_common_parameters(self)
        ros_config = load_yaml(params["ros_config"])
        validate_ros_config(ros_config)
        self._measurement_context = measurement_context_for_mode(params["mode"])
        self._path = Path(params["results_dir"]) / "runtime" / "validation.csv"
        self.create_subscription(
            OccupancyGrid3D, ros_config["occupancy_topic"], self._on_occupancy, 10
        )
        self.create_subscription(
            OccupancyFlow3D, ros_config["flow_topic"], self._on_flow, 10
        )

    def _on_occupancy(self, message: OccupancyGrid3D) -> None:
        self._check("occupancy", message, occupancy_from_msg)

    def _on_flow(self, message: OccupancyFlow3D) -> None:
        self._check("flow", message, flow_from_msg)

    def _check(self, kind: str, message, converter) -> None:
        try:
            prediction = converter(message)
            passed = True
            detail = "valid"
        except (PredictionValidationError, ValueError, TypeError) as exc:
            prediction = None
            passed = False
            detail = str(exc)
            self.get_logger().error(f"{kind} validation failed: {detail}")
        append_csv(
            self._path,
            FIELDS,
            {
                "timestamp_sec": int(message.header.stamp.sec),
                "timestamp_nanosec": int(message.header.stamp.nanosec),
                "kind": kind,
                "passed": str(passed).lower(),
                "is_mock": str(bool(message.is_mock)).lower(),
                "prediction_source": message.prediction_source,
                "detail": detail,
                "measurement_context": self._measurement_context,
            },
        )
        if prediction is not None:
            self.get_logger().info(
                f"Validated {kind} mock={prediction.is_mock} source={prediction.source}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PredictionValidator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
