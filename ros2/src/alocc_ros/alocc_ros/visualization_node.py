"""Publish RViz markers and write SVG slices for occupancy predictions.

SVG slices stay index-based and are written for every occupancy message.
RViz cubes use one CUBE_LIST marker per semantic class. Free space is omitted.
The occupancy publisher is a different process. Marker construction runs on a
worker thread. The subscription keeps only the newest sample so a slow draw
does not queue every grid behind the publisher.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import rclpy
from alocc_interfaces.msg import OccupancyFlow3D, OccupancyGrid3D
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray

from alocc_ros.config_loader import load_yaml, validate_ros_config
from alocc_ros.message_conversion import flow_from_msg, occupancy_from_msg
from alocc_ros.node_utils import declare_common_parameters
from alocc_ros.occupancy_geometry import (
    CLASS_COLORS,
    color_hex,
    color_rgb,
    marker_options_from_config,
    voxel_center,
)
from alocc_ros.prediction_types import index_to_xyz
from alocc_ros.visualization_render import render_flow_slice_svg, render_occupancy_slice_svg

# Reliable delivery is required for a 640000-voxel occupancy message.
# Best-effort UDP dropped those messages. Depth 1 keeps only the newest sample.
_LISTEN_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


@dataclass(frozen=True)
class OccupancyMarkerBuild:
    markers: list
    rendered: int
    skipped_free: int
    skipped_stride: int
    skipped_class: int
    truncated: bool


class VisualizationNode(Node):
    def __init__(self):
        super().__init__("visualization_node")
        params = declare_common_parameters(self)
        ros_config = load_yaml(params["ros_config"])
        validate_ros_config(ros_config)
        self._frame_id = ros_config["frame_id"]
        self._options = marker_options_from_config(ros_config)
        self._min_period_sec = float((ros_config.get("visualization") or {}).get("min_period_sec", 0.0))
        self._output_dir = Path(params["results_dir"]) / "visualization"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._marker_pub = self.create_publisher(MarkerArray, ros_config["marker_topic"], 10)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest_occupancy = None
        self._latest_flow = None
        self._pending = False
        self._ready = None
        self._last_marker_time = None
        self._legend_logged = False
        self._worker = threading.Thread(
            target=self._build_loop, name="alocc-markers", daemon=True
        )
        self._worker.start()
        self.create_timer(0.2, self._flush_markers)
        self.create_subscription(
            OccupancyGrid3D,
            ros_config["occupancy_topic"],
            self._on_occupancy,
            _LISTEN_QOS,
        )
        self.create_subscription(
            OccupancyFlow3D,
            ros_config["flow_topic"],
            self._on_flow,
            _LISTEN_QOS,
        )

    def _on_occupancy(self, message: OccupancyGrid3D) -> None:
        prediction = occupancy_from_msg(message)
        (self._output_dir / "occupancy_slice.svg").write_text(
            render_occupancy_slice_svg(prediction), encoding="utf-8"
        )
        with self._lock:
            self._latest_occupancy = prediction
            self._pending = True
        self.get_logger().info(
            f"Queued occupancy visualization voxels={prediction.spatial.voxel_count()} "
            f"geometry_confirmed={prediction.spatial.geometry_confirmed}"
        )

    def _on_flow(self, message: OccupancyFlow3D) -> None:
        prediction = flow_from_msg(message)
        (self._output_dir / "flow_slice.svg").write_text(
            render_flow_slice_svg(prediction), encoding="utf-8"
        )
        with self._lock:
            self._latest_flow = prediction
            self._pending = True

    def _build_loop(self) -> None:
        """Build marker messages away from the occupancy subscription callback."""
        while not self._stop.is_set():
            with self._lock:
                if not self._pending:
                    occupancy = None
                    flow = None
                else:
                    self._pending = False
                    occupancy = self._latest_occupancy
                    flow = self._latest_flow
            if occupancy is None and flow is None:
                self._stop.wait(0.05)
                continue
            now = monotonic()
            if (
                self._min_period_sec > 0.0
                and self._last_marker_time is not None
                and (now - self._last_marker_time) < self._min_period_sec
            ):
                continue
            array, summary, info = self._make_markers(occupancy, flow)
            with self._lock:
                self._ready = (array, summary, info)
            self._last_marker_time = now

    def _flush_markers(self) -> None:
        with self._lock:
            ready = self._ready
            self._ready = None
        if ready is None:
            return
        array, summary, info = ready
        (self._output_dir / "marker_summary.txt").write_text(summary + "\n", encoding="utf-8")
        self._marker_pub.publish(array)
        if info:
            self.get_logger().info(info)
            self._log_legend_once()

    def _make_markers(self, occupancy, flow):
        array = MarkerArray()
        summary = "rendered=0 skipped_free=0 skipped_stride=0 skipped_class=0 truncated=false"
        info = ""
        if occupancy is not None:
            spatial = occupancy.spatial
            namespace = "mock_occupancy" if occupancy.is_mock else "occupancy"
            array.markers.append(_delete_all(self._frame_id, namespace))
            if not spatial.geometry_confirmed:
                summary = "skipped_reason=geometry_unconfirmed rendered=0 truncated=false"
                info = (
                    "Skipping RViz occupancy cubes because geometry is unconfirmed. "
                    "Index positions are not meters."
                )
            else:
                built = build_occupancy_markers(occupancy, self._frame_id, self._options)
                array.markers.extend(built.markers)
                selected = (
                    "all_except_free"
                    if self._options.classes is None
                    else ",".join(sorted(self._options.classes))
                )
                summary = (
                    f"frame={self._frame_id} rendered={built.rendered} "
                    f"skipped_free={built.skipped_free} skipped_stride={built.skipped_stride} "
                    f"skipped_class={built.skipped_class} truncated={str(built.truncated).lower()} "
                    f"stride={self._options.downsample_stride} max_voxels={self._options.max_voxels} "
                    f"classes={selected}"
                )
                info = (
                    f"Occupancy markers: {built.rendered} voxels, "
                    f"skipped {built.skipped_free} free. {summary}"
                )
        if flow is not None:
            flow_count = flow.spatial.voxel_count()
            array.markers.append(_delete_all(self._frame_id, "mock_flow"))
            if flow_count > 4096:
                if info:
                    info += " "
                info += (
                    "Skipping flow arrows for a large grid. "
                    "The motion-flow head is outside the current occupancy milestone."
                )
            else:
                array.markers.extend(build_flow_markers(flow, self._frame_id))
        return array, summary, info

    def _log_legend_once(self) -> None:
        if self._legend_logged:
            return
        self._legend_logged = True
        legend = ", ".join(
            f"{name}={color_hex(name)}" for name in CLASS_COLORS if name != "free"
        )
        self.get_logger().info(
            "Class colors, free is not drawn: " + legend + ". Frame " + self._frame_id + " is ego."
        )


def build_occupancy_markers(prediction, frame_id: str, options) -> OccupancyMarkerBuild:
    """Group occupied voxels into one CUBE_LIST per class.

    Free space is omitted. `others` is not free and is drawn unless a class
    filter excludes it. A stride greater than 1 keeps voxels whose i, j, and k
    are all divisible by that stride, which drops occupied voxels on purpose.
    A positive max_voxels stops in x-fastest order, so later z slices are the
    ones left out. max_voxels 0 and stride 1 keep every selected voxel.
    """
    buckets: dict[int, list[Point]] = {}
    rendered = 0
    skipped_free = 0
    skipped_stride = 0
    skipped_class = 0
    truncated = False
    stride = options.downsample_stride
    spatial = prediction.spatial
    for index, label in enumerate(prediction.semantic_labels):
        class_name = prediction.class_names[label]
        if class_name == options.free_class:
            skipped_free += 1
            continue
        if options.classes is not None and class_name not in options.classes:
            skipped_class += 1
            continue
        i, j, k = index_to_xyz(index, spatial)
        if stride > 1 and (i % stride or j % stride or k % stride):
            skipped_stride += 1
            continue
        if options.max_voxels and rendered >= options.max_voxels:
            truncated = True
            break
        x, y, z = voxel_center(i, j, k, spatial)
        point = Point()
        point.x = x
        point.y = y
        point.z = z
        buckets.setdefault(label, []).append(point)
        rendered += 1
    markers = []
    namespace = "mock_occupancy" if prediction.is_mock else "occupancy"
    for label in sorted(buckets):
        class_name = prediction.class_names[label]
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp.sec = prediction.timestamp_sec
        marker.header.stamp.nanosec = prediction.timestamp_nanosec
        marker.ns = namespace
        marker.id = int(label)
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = spatial.resolution_x
        marker.scale.y = spatial.resolution_y
        marker.scale.z = spatial.resolution_z
        red, green, blue = color_rgb(class_name, label)
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 0.9
        marker.points = buckets[label]
        markers.append(marker)
    return OccupancyMarkerBuild(
        markers=markers,
        rendered=rendered,
        skipped_free=skipped_free,
        skipped_stride=skipped_stride,
        skipped_class=skipped_class,
        truncated=truncated,
    )


def build_flow_markers(prediction, frame_id: str) -> list[Marker]:
    markers = []
    spatial = prediction.spatial
    marker_id = 0
    for index in range(spatial.voxel_count()):
        i, j, k = index_to_xyz(index, spatial)
        vx, vy, vz = prediction.flow_vectors[index * 3 : index * 3 + 3]
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp.sec = prediction.timestamp_sec
        marker.header.stamp.nanosec = prediction.timestamp_nanosec
        marker.ns = "mock_flow"
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        start = Point()
        start.x, start.y, start.z = voxel_center(i, j, k, spatial)
        end = Point()
        end.x = start.x + vx
        end.y = start.y + vy
        end.z = start.z + vz
        marker.points = [start, end]
        marker.scale.x = 0.02
        marker.scale.y = 0.04
        marker.color.r = 0.5
        marker.color.g = 0.85
        marker.color.b = 1.0
        marker.color.a = 0.9
        markers.append(marker)
        marker_id += 1
    return markers


def _delete_all(frame_id: str, namespace: str) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = namespace
    marker.id = 0
    marker.action = Marker.DELETEALL
    return marker


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
