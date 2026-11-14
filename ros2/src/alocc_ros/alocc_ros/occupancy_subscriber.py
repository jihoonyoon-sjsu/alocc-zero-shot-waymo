"""Receive occupancy predictions and record that they arrived."""

from __future__ import annotations

from pathlib import Path

import rclpy
from alocc_interfaces.msg import OccupancyGrid3D
from rclpy.node import Node

from alocc_ros.config_loader import load_yaml, validate_ros_config
from alocc_ros.metrics import append_csv
from alocc_ros.node_utils import declare_common_parameters
from alocc_ros.prediction_types import measurement_context_for_mode

FIELDS = [
    "timestamp_sec",
    "timestamp_nanosec",
    "sample_index",
    "dim_x",
    "dim_y",
    "dim_z",
    "is_mock",
    "prediction_source",
    "label_count",
    "label_sha256",
    "flow_supplied",
    "measurement_context",
]


class OccupancySubscriber(Node):
    def __init__(self):
        super().__init__("occupancy_subscriber")
        params = declare_common_parameters(self)
        ros_config = load_yaml(params["ros_config"])
        validate_ros_config(ros_config)
        self._measurement_context = measurement_context_for_mode(params["mode"])
        self._path = Path(params["results_dir"]) / "runtime" / "received_occupancy.csv"
        self._count = 0
        self.create_subscription(
            OccupancyGrid3D,
            ros_config["occupancy_topic"],
            self._on_message,
            10,
        )

    def _on_message(self, message: OccupancyGrid3D) -> None:
        self._count += 1
        append_csv(
            self._path,
            FIELDS,
            {
                "timestamp_sec": int(message.header.stamp.sec),
                "timestamp_nanosec": int(message.header.stamp.nanosec),
                "sample_index": int(message.sample_index),
                "dim_x": int(message.dim_x),
                "dim_y": int(message.dim_y),
                "dim_z": int(message.dim_z),
                "is_mock": str(bool(message.is_mock)).lower(),
                "prediction_source": message.prediction_source,
                "label_count": len(message.semantic_labels),
                "label_sha256": _sha256(message.semantic_labels),
                "flow_supplied": str(bool(message.flow_supplied)).lower(),
                "measurement_context": self._measurement_context,
            },
        )
        self.get_logger().info(
            f"Received occupancy #{self._count} mock={message.is_mock} "
            f"source={message.prediction_source} voxels={len(message.semantic_labels)}"
        )


def _sha256(labels) -> str:
    import hashlib

    return hashlib.sha256(bytes(labels)).hexdigest()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OccupancySubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
