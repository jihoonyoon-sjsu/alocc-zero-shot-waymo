import time

import pytest

pytest.importorskip("alocc_interfaces.msg")
from alocc_interfaces.msg import OccupancyFlow3D, OccupancyGrid3D

from alocc_ros.config_loader import load_yaml
from alocc_ros.connectors.mock import generate_mock_pair
from alocc_ros.connectors.ros_subscription import RosSubscriptionConnector
from alocc_ros.message_conversion import flow_to_msg, occupancy_to_msg


def _spin_until(rclpy, nodes, predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
    return False


def test_occupancy_and_flow_publish_and_subscribe(ros_context, root):
    rclpy = ros_context
    model = load_yaml(root / "config" / "model.yaml")
    occupancy, flow = generate_mock_pair(model, timestamp_sec=20, timestamp_nanosec=21)
    publisher = rclpy.create_node("alocc_test_publisher")
    subscriber = rclpy.create_node("alocc_test_subscriber")
    received = {"occupancy": [], "flow": []}
    occ_topic = "/alocc/test/occupancy"
    flow_topic = "/alocc/test/flow"
    try:
        occ_pub = publisher.create_publisher(OccupancyGrid3D, occ_topic, 10)
        flow_pub = publisher.create_publisher(OccupancyFlow3D, flow_topic, 10)
        subscriber.create_subscription(
            OccupancyGrid3D, occ_topic, lambda msg: received["occupancy"].append(msg), 10
        )
        subscriber.create_subscription(
            OccupancyFlow3D, flow_topic, lambda msg: received["flow"].append(msg), 10
        )
        occ_msg = occupancy_to_msg(occupancy, "alocc_placeholder")
        flow_msg = flow_to_msg(flow, "alocc_placeholder")

        def publish_and_check():
            occ_pub.publish(occ_msg)
            flow_pub.publish(flow_msg)
            return received["occupancy"] and received["flow"]

        assert _spin_until(rclpy, [publisher, subscriber], publish_and_check)
        assert received["occupancy"][0].is_mock is True
        assert list(received["occupancy"][0].semantic_labels) == list(occupancy.semantic_labels)
        assert received["flow"][0].prediction_source == "mock"
        assert len(received["flow"][0].flow_vectors) == len(flow.flow_vectors)
    finally:
        publisher.destroy_node()
        subscriber.destroy_node()


def test_ros_connector_converts_received_pair(ros_context, root):
    rclpy = ros_context
    model = load_yaml(root / "config" / "model.yaml")
    ros_config = load_yaml(root / "config" / "ros2.yaml")
    model["ros_input"] = {
        "occupancy_topic": "/alocc/test/input_occupancy",
        "flow_topic": "/alocc/test/input_flow",
        "wait_timeout_sec": 5.0,
        "sync_tolerance_ns": 0,
        "max_queue": 16,
    }
    occupancy, flow = generate_mock_pair(model, timestamp_sec=30, timestamp_nanosec=1)
    source = rclpy.create_node("alocc_test_ros_source")
    listener = rclpy.create_node("alocc_test_ros_listener")
    try:
        connector = RosSubscriptionConnector(model, ros_config, listener)
        occ_pub = source.create_publisher(OccupancyGrid3D, model["ros_input"]["occupancy_topic"], 10)
        flow_pub = source.create_publisher(OccupancyFlow3D, model["ros_input"]["flow_topic"], 10)
        occ_msg = occupancy_to_msg(occupancy, "alocc_placeholder")
        flow_msg = flow_to_msg(flow, "alocc_placeholder")

        def publish():
            occ_pub.publish(occ_msg)
            flow_pub.publish(flow_msg)
            return connector.ready()

        assert _spin_until(rclpy, [source, listener], publish)
        loaded_occ, loaded_flow = connector.load()
        assert loaded_occ.source == "ros"
        assert loaded_flow.source == "ros"
        assert loaded_occ.semantic_labels == occupancy.semantic_labels
    finally:
        source.destroy_node()
        listener.destroy_node()
