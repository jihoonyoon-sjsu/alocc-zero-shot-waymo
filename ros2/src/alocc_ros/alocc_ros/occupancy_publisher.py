"""Publish occupancy predictions on /alocc/occupancy by default."""

from __future__ import annotations

import rclpy
from alocc_interfaces.msg import OccupancyGrid3D
from rclpy.node import Node

from alocc_ros.config_loader import load_yaml, validate_mode, validate_ros_config
from alocc_ros.connectors import create_connector
from alocc_ros.message_conversion import occupancy_to_msg
from alocc_ros.node_utils import declare_common_parameters


class OccupancyPublisher(Node):
    def __init__(self):
        super().__init__("occupancy_publisher")
        params = declare_common_parameters(self)
        self._model_config = load_yaml(params["model_config"])
        self._ros_config = load_yaml(params["ros_config"])
        validate_ros_config(self._ros_config)
        validate_mode(params["mode"], self._model_config)
        self._frame_id = self._ros_config["frame_id"]
        self._publisher = self.create_publisher(
            OccupancyGrid3D,
            self._ros_config["occupancy_topic"],
            10,
        )
        self._connector = create_connector(
            params["mode"], self._model_config, self._ros_config, self
        )
        period = 1.0 / float(self._ros_config["publish_rate_hz"])
        self.create_timer(period, self._publish_once)
        self.get_logger().info(
            f"Occupancy publisher using {params['mode']} data on {self._ros_config['occupancy_topic']}"
        )

    def publish_prediction(self, prediction) -> None:
        self._publisher.publish(occupancy_to_msg(prediction, self._frame_id))

    def _publish_once(self) -> None:
        from alocc_ros.connectors.base import ConnectorUnavailable
        from alocc_ros.prediction_types import PredictionValidationError

        try:
            occupancy, _flow = self._connector.load()
        except (ConnectorUnavailable, PredictionValidationError) as exc:
            self.get_logger().warning(str(exc))
            return
        self.publish_prediction(occupancy)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OccupancyPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
