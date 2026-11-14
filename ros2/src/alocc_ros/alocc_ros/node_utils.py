"""Small helpers shared by the rclpy nodes."""

from __future__ import annotations

from pathlib import Path

import rclpy
import yaml


def spin_node(node_factory):
    rclpy.init()
    node = node_factory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def load_yaml_mapping(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def declare_common_parameters(node) -> dict:
    model_config = node.declare_parameter("model_config", "").get_parameter_value().string_value
    ros_config = node.declare_parameter("ros_config", "").get_parameter_value().string_value
    mode = node.declare_parameter("mode", "mock").get_parameter_value().string_value
    results_dir = node.declare_parameter("results_dir", "results").get_parameter_value().string_value
    return {
        "model_config": model_config,
        "ros_config": ros_config,
        "mode": mode,
        "results_dir": results_dir,
    }
