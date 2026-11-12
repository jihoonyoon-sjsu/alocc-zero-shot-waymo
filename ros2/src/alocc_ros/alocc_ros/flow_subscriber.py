"""Receive flow predictions and record that they arrived."""

from __future__ import annotations

from pathlib import Path

import rclpy
from alocc_interfaces.msg import OccupancyFlow3D
from rclpy.node import Node

from alocc_ros.config_loader import load_yaml, validate_ros_config
from alocc_ros.metrics import append_csv
from alocc_ros.node_utils import declare_common_parameters
from alocc_ros.prediction_types import measurement_context_for_mode

FIELDS = [
    "timestamp_sec",
    "timestamp_nanosec",
    "is_mock",
    "prediction_source",
    "vector_count",
    "measurement_context",
]


class FlowSubscriber(Node):
    def __init__(self):
        super().__init__("flow_subscriber")
        params = declare_common_parameters(self)
        ros_config = load_yaml(params["ros_config"])
        validate_ros_config(ros_config)
        self._measurement_context = measurement_context_for_mode(params["mode"])
        self._path = Path(params["results_dir"]) / "runtime" / "received_flow.csv"
        self._count = 0
        self.create_subscription(
            OccupancyFlow3D,
            ros_config["flow_topic"],
            self._on_message,
            10,
        )

    def _on_message(self, message: OccupancyFlow3D) -> None:
        self._count += 1
        append_csv(
            self._path,
            FIELDS,
            {
                "timestamp_sec": int(message.header.stamp.sec),
                "timestamp_nanosec": int(message.header.stamp.nanosec),
                "is_mock": str(bool(message.is_mock)).lower(),
                "prediction_source": message.prediction_source,
                "vector_count": len(message.flow_vectors),
                "measurement_context": self._measurement_context,
            },
        )
        self.get_logger().info(
            f"Received flow #{self._count} mock={message.is_mock} "
            f"source={message.prediction_source} values={len(message.flow_vectors)}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FlowSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
