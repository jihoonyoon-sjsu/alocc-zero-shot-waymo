"""Publish synchronized Waymo scene 0 images, camera info, and occupancy.

Message header stamps come from ``frames.json`` ``timestamp_us``. The clock
that spaces the frames is monotonic and is not written into the messages.
No flow topic is published.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from alocc_interfaces.msg import OccupancyGrid3D
from alocc_ros.waymo_scene0_dataset import (
    EGO_FRAME_ID,
    OCCUPANCY_TOPIC,
    WaymoSceneError,
    camera_info_topic,
    decode_rgb_image,
    image_topic,
    load_replay_scene,
            loop_wait_seconds,
            remaining_wait,
            ros_stamp,
            select_frames,
            wait_seconds,
)
from alocc_ros.waymo_scene0_messages import build_camera_info, build_image, build_occupancy

# Best-effort transport dropped large occupancy messages on this machine.
# Camera images are larger still, so images, camera info, and occupancy all
# use reliable delivery and a short queue.
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)


def _default_model_config() -> str:
    """Find config/model.yaml from this file's project directory."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / "model.yaml"
        if candidate.is_file():
            return str(candidate)
    return "config/model.yaml"


class WaymoScene0ReplayNode(Node):
    def __init__(self):
        super().__init__("waymo_scene0_replay_node")
        dataset_path = str(
            Path(
                self.declare_parameter(
                    "dataset_path", "~/alocc-data/waymo_scene0"
                ).get_parameter_value().string_value
            ).expanduser()
        )
        model_config = str(
            Path(
                self.declare_parameter(
                    "model_config", _default_model_config()
                ).get_parameter_value().string_value
            ).expanduser()
        )
        self._replay_rate = self.declare_parameter("replay_rate", 1.0).get_parameter_value().double_value
        self._loop = self.declare_parameter("loop", False).get_parameter_value().bool_value
        start_frame = self.declare_parameter("start_frame", 0).get_parameter_value().integer_value
        end_frame = self.declare_parameter("end_frame", 39).get_parameter_value().integer_value
        self._publish_images = self.declare_parameter(
            "publish_camera_images", True
        ).get_parameter_value().bool_value
        self._publish_info = self.declare_parameter(
            "publish_camera_info", True
        ).get_parameter_value().bool_value
        self._publish_occupancy = self.declare_parameter(
            "publish_occupancy", True
        ).get_parameter_value().bool_value
        try:
            self._scene = load_replay_scene(dataset_path, model_config)
            self._frames = select_frames(self._scene, int(start_frame), int(end_frame))
            if self._replay_rate <= 0.0:
                raise WaymoSceneError("replay_rate must be positive")
        except (WaymoSceneError, Exception) as exc:
            self.get_logger().error(str(exc))
            raise
        self._image_pubs = {}
        self._info_pubs = {}
        if self._publish_images:
            for camera in self._scene.cameras:
                self._image_pubs[camera.name] = self.create_publisher(
                    Image, image_topic(camera.topic_suffix), RELIABLE_QOS
                )
        if self._publish_info:
            for camera in self._scene.cameras:
                self._info_pubs[camera.name] = self.create_publisher(
                    CameraInfo, camera_info_topic(camera.topic_suffix), RELIABLE_QOS
                )
        self._occupancy_pub = None
        if self._publish_occupancy:
            self._occupancy_pub = self.create_publisher(
                OccupancyGrid3D, OCCUPANCY_TOPIC, RELIABLE_QOS
            )
        self._cancel = threading.Event()
        self._done = threading.Event()
        self.failed = False
        self.get_logger().info(
            "Waymo scene 0 occupancy source: "
            f"{self._scene.prediction_path} "
            f"frames={len(self._frames)} rate={self._replay_rate} loop={self._loop}"
        )
        self.get_logger().info(
            "Camera calibration provides width, height, and K. "
            "D, R, and P are not in calibration.json and are left unset."
        )
        self._thread = threading.Thread(target=self._replay, name="waymo-scene0-replay", daemon=True)
        self._thread.start()

    def _replay(self) -> None:
        try:
            self.get_logger().info("Waiting 1.0s so subscribers can connect")
            if not self._sleep(1.0):
                return
            index = 0
            while not self._cancel.is_set():
                if index >= len(self._frames):
                    if not self._loop:
                        break
                    index = 0
                frame = self._frames[index]
                started = time.monotonic()
                self._publish_frame(frame)
                index += 1
                if index < len(self._frames):
                    delay = wait_seconds(
                        frame.timestamp_us, self._frames[index].timestamp_us, self._replay_rate
                    )
                elif self._loop:
                    delay = loop_wait_seconds(self._frames, self._replay_rate)
                else:
                    break
                remaining = remaining_wait(delay, time.monotonic() - started)
                if remaining > 0.0 and not self._sleep(remaining):
                    return
                if remaining == 0.0 and delay > 0.0:
                    self.get_logger().warning(
                        f"Frame {frame.frame_idx} took longer than the {delay:.3f}s dataset interval"
                    )
            self.get_logger().info(
                f"Waymo scene 0 replay finished frames={len(self._frames)} loop={self._loop}"
            )
            self._sleep(2.0)
        except Exception as exc:
            self.failed = True
            self.get_logger().error(f"Waymo scene 0 replay failed: {exc}")
        finally:
            self._done.set()

    def _sleep(self, delay: float) -> bool:
        deadline = time.monotonic() + max(delay, 0.0)
        while not self._cancel.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return True
            time.sleep(min(0.05, remaining))
        return False

    def _publish_frame(self, frame) -> None:
        sec, nanosec = ros_stamp(frame.timestamp_us)
        if self._publish_images or self._publish_info:
            for camera in self._scene.cameras:
                if self._publish_images:
                    height, width, data = decode_rgb_image(frame.image_paths[camera.name])
                    if height != camera.height or width != camera.width:
                        raise WaymoSceneError(
                            f"{camera.name} image is {width}x{height}, "
                            f"calibration is {camera.width}x{camera.height}"
                        )
                    self._image_pubs[camera.name].publish(
                        build_image(camera.name, sec, nanosec, height, width, data)
                    )
                if self._publish_info:
                    self._info_pubs[camera.name].publish(build_camera_info(camera, sec, nanosec))
        if self._occupancy_pub is not None:
            self._occupancy_pub.publish(build_occupancy(self._scene, frame))
        self.get_logger().info(
            "Replayed "
            f"order={frame.order} frame_idx={frame.frame_idx} "
            f"timestamp_us={frame.timestamp_us} stamp={sec}.{nanosec:09d} frame_id={EGO_FRAME_ID}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = WaymoScene0ReplayNode()
    except Exception:
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(1)
    try:
        while rclpy.ok() and not node._done.is_set():
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node._cancel.set()
    node._cancel.set()
    node._thread.join(timeout=5.0)
    failed = node.failed
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(1 if failed else 0)
