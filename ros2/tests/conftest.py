import os
from pathlib import Path

import pytest

os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
os.environ.setdefault("ROS_DOMAIN_ID", "42")


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "config" / "model.yaml").is_file():
            return parent
    raise RuntimeError("Could not find the ROS 2 project root")


@pytest.fixture(scope="session")
def root() -> Path:
    return repo_root()


@pytest.fixture(scope="session")
def ros_context():
    rclpy = pytest.importorskip("rclpy")
    if not rclpy.ok():
        rclpy.init()
    yield rclpy
    if rclpy.ok():
        rclpy.shutdown()
