"""Record scene 0 replay headers from the topics themselves.

A frame counts as synchronized only after all five images, all five camera
info messages, and the occupancy message have been received, and every header
stamp converts back to that frame's dataset ``timestamp_us``.
"""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from alocc_interfaces.msg import OccupancyGrid3D
from alocc_ros.waymo_scene0_dataset import (
    CAMERA_NAMES,
    CAMERA_SPECS,
    CSV_COLUMNS,
    EGO_FRAME_ID,
    OCCUPANCY_TOPIC,
    camera_info_topic,
    image_topic,
    parse_metadata,
    ros_stamp,
    validation_row,
)

RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


@dataclass
class PartialFrame:
    images: dict = field(default_factory=dict)
    infos: dict = field(default_factory=dict)
    occupancy: dict | None = None


class WaymoScene0ReplayCheckNode(Node):
    def __init__(self):
        super().__init__("waymo_scene0_replay_check_node")
        self._expected = int(self.declare_parameter("expected_frames", 40).value)
        self._csv_path = Path(
            self.declare_parameter(
                "validation_csv",
                "~/alocc-data/waymo_scene0/reports/replay_validation.csv",
            ).value
        ).expanduser()
        timeout_sec = float(self.declare_parameter("timeout_sec", 180.0).value)
        self._deadline = time.monotonic() + timeout_sec
        self._partials = {}
        self._rows = []
        self._closed = set()
        self.failed = False
        self._finished = False
        for _name, suffix in CAMERA_SPECS:
            self.create_subscription(Image, image_topic(suffix), self._on_image, RELIABLE_QOS)
            self.create_subscription(
                CameraInfo, camera_info_topic(suffix), self._on_info, RELIABLE_QOS
            )
        self.create_subscription(OccupancyGrid3D, OCCUPANCY_TOPIC, self._on_occupancy, RELIABLE_QOS)
        self.create_timer(0.2, self._watch)
        self.get_logger().info(
            f"Waiting for {self._expected} synchronized scene 0 frames -> {self._csv_path}"
        )

    def _stamp_key(self, stamp) -> tuple[int, int]:
        return int(stamp.sec), int(stamp.nanosec)

    def _partial(self, key) -> PartialFrame:
        if key not in self._partials:
            self._partials[key] = PartialFrame()
        return self._partials[key]

    def _on_image(self, message: Image) -> None:
        key = self._stamp_key(message.header.stamp)
        if key in self._closed:
            return
        self._partial(key).images[message.header.frame_id] = (
            key[0],
            key[1],
            int(message.width),
            int(message.height),
            str(message.encoding),
        )
        self._maybe_close(key)

    def _on_info(self, message: CameraInfo) -> None:
        key = self._stamp_key(message.header.stamp)
        if key in self._closed:
            return
        self._partial(key).infos[message.header.frame_id] = (
            key[0],
            key[1],
            int(message.width),
            int(message.height),
            tuple(float(value) for value in message.k),
        )
        self._maybe_close(key)

    def _on_occupancy(self, message: OccupancyGrid3D) -> None:
        key = self._stamp_key(message.header.stamp)
        if key in self._closed:
            return
        self._partial(key).occupancy = {
            "sec": key[0],
            "nanosec": key[1],
            "frame_id": str(message.header.frame_id),
            "sample_index": int(message.sample_index),
            "metadata": str(message.sample_metadata),
            "shape": f"{int(message.dim_x)}x{int(message.dim_y)}x{int(message.dim_z)}",
            "flow_supplied": bool(message.flow_supplied),
            "timestamp_available": bool(message.timestamp_available),
            "class_count": len(message.class_names),
            "free_name": str(message.class_names[-1]) if message.class_names else "",
            "label_count": len(message.semantic_labels),
        }
        self._maybe_close(key)

    def _maybe_close(self, key) -> None:
        partial = self._partials.get(key)
        if partial is None or partial.occupancy is None:
            return
        if len(partial.images) < len(CAMERA_NAMES) or len(partial.infos) < len(CAMERA_NAMES):
            return
        self._rows.append(self._row_for(key, partial))
        self._closed.add(key)
        self._partials.pop(key, None)
        self.get_logger().info(
            f"Received synchronized frame {len(self._rows)}/{self._expected} "
            f"stamp={key[0]}.{key[1]:09d}"
        )

    def _row_for(self, key, partial: PartialFrame) -> dict[str, str]:
        occupancy = partial.occupancy
        meta = parse_metadata(occupancy["metadata"])
        reasons = []
        try:
            frame_idx = int(meta["frame_idx"])
            timestamp_us = int(meta["timestamp_us"])
            order = int(meta["order"])
        except (KeyError, ValueError):
            frame_idx = -1
            timestamp_us = -1
            order = -1
            reasons.append("metadata")
        expected = None
        if timestamp_us >= 0:
            expected = ros_stamp(timestamp_us)
        stamps = [key]
        stamps.extend((item[0], item[1]) for item in partial.images.values())
        stamps.extend((item[0], item[1]) for item in partial.infos.values())
        sync = expected is not None and all(stamp == expected for stamp in stamps)
        if not sync:
            reasons.append("timestamp")
        if set(partial.images) != set(CAMERA_NAMES) or set(partial.infos) != set(CAMERA_NAMES):
            reasons.append("cameras")
        if any(item[4] != "rgb8" for item in partial.images.values()):
            reasons.append("encoding")
        if occupancy["frame_id"] != EGO_FRAME_ID:
            reasons.append("frame_id")
        if occupancy["shape"] != "200x200x16" or occupancy["label_count"] != 200 * 200 * 16:
            reasons.append("shape")
        if occupancy["flow_supplied"] or not occupancy["timestamp_available"]:
            reasons.append("occupancy_flags")
        if occupancy["class_count"] != 18 or occupancy["free_name"] != "free":
            reasons.append("classes")
        if occupancy["sample_index"] != order:
            reasons.append("sample_index")
        status = "ok" if not reasons else "error:" + ",".join(reasons)
        return validation_row(
            frame_idx,
            timestamp_us,
            occupancy["sample_index"],
            len(partial.images),
            len(partial.infos),
            1,
            occupancy["shape"],
            sync,
            status,
        )

    def _watch(self) -> None:
        if self._finished:
            return
        if len(self._rows) >= self._expected:
            self._finish(True)
            return
        if time.monotonic() >= self._deadline:
            self._finish(False)

    def _finish(self, complete: bool) -> None:
        if self._finished:
            return
        self._finished = True
        for key, partial in list(self._partials.items()):
            occupancy = partial.occupancy
            meta = parse_metadata(occupancy["metadata"]) if occupancy else {}
            self._rows.append(
                validation_row(
                    int(meta.get("frame_idx", -1)),
                    int(meta.get("timestamp_us", -1)),
                    int(occupancy["sample_index"]) if occupancy else -1,
                    len(partial.images),
                    len(partial.infos),
                    1 if occupancy else 0,
                    occupancy["shape"] if occupancy else "",
                    False,
                    "incomplete",
                )
            )
        self._rows.sort(key=lambda row: int(row["frame_idx"]))
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)
        ok_rows = [row for row in self._rows if row["status"] == "ok"]
        self.failed = not (complete and len(ok_rows) == self._expected)
        self.get_logger().info(
            f"Wrote {self._csv_path} rows={len(self._rows)} ok={len(ok_rows)} "
            f"expected={self._expected} failed={self.failed}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WaymoScene0ReplayCheckNode()
    try:
        while rclpy.ok() and not node._finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        if not node._finished:
            node._finish(False)
    failed = node.failed
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(1 if failed else 0)
