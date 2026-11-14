"""Subscribe to framework occupancy and flow messages.

The node executor must be spinning so subscription callbacks can store
messages. `load()` does not spin. It pairs one occupancy message with one
flow message only when their frame ids match and their timestamps fall
within `ros_input.sync_tolerance_ns`. A missing side stays buffered until
`wait_timeout_sec` elapses or `max_queue` drops it.
"""

from __future__ import annotations

import threading
import time

from alocc_ros.connectors.base import ConnectorUnavailable, PredictionConnector
from alocc_ros.prediction_types import PredictionValidationError, stamp_to_ns, validate_prediction_pair


class RosSubscriptionConnector(PredictionConnector):
    name = "ros"

    def __init__(self, model_config: dict, ros_config: dict, node):
        del ros_config  # topics and sync settings live in the model config
        if node is None:
            raise ConnectorUnavailable(
                "ROS subscription connector needs a running rclpy node. "
                "Person 2 is not publishing predictions yet."
            )
        from alocc_interfaces.msg import OccupancyFlow3D, OccupancyGrid3D

        ros_input = model_config.get("ros_input") or {}
        occupancy_topic = str(ros_input.get("occupancy_topic") or "")
        flow_topic = str(ros_input.get("flow_topic") or "")
        if not occupancy_topic or not flow_topic:
            raise ConnectorUnavailable(
                "ros_input topics are empty. Choose input topic names with Person 2."
            )
        tolerance = int(ros_input.get("sync_tolerance_ns", 0))
        max_queue = int(ros_input.get("max_queue", 16))
        timeout = float(ros_input.get("wait_timeout_sec", 1.0))
        if tolerance < 0:
            raise ConnectorUnavailable("ros_input.sync_tolerance_ns must be >= 0")
        if max_queue < 1:
            raise ConnectorUnavailable("ros_input.max_queue must be >= 1")
        if timeout < 0.0:
            raise ConnectorUnavailable("ros_input.wait_timeout_sec must be >= 0")
        self._node = node
        self._timeout = timeout
        self._tolerance_ns = tolerance
        self._max_queue = max_queue
        self._occupancy_topic = occupancy_topic
        self._flow_topic = flow_topic
        self._occupancy: dict[tuple[int, int], tuple[float, object]] = {}
        self._flow: dict[tuple[int, int], tuple[float, object]] = {}
        self._lock = threading.Lock()
        node.create_subscription(OccupancyGrid3D, occupancy_topic, self._on_occupancy, 10)
        node.create_subscription(OccupancyFlow3D, flow_topic, self._on_flow, 10)

    def ready(self) -> bool:
        with self._lock:
            return self._find_pair() is not None

    def buffered_counts(self) -> tuple[int, int]:
        with self._lock:
            return len(self._occupancy), len(self._flow)

    def buffered_stamps(self) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        with self._lock:
            return tuple(self._occupancy), tuple(self._flow)

    def load(self) -> tuple:
        from alocc_ros.message_conversion import flow_from_msg, occupancy_from_msg

        with self._lock:
            expired = self._expire()
            pair = self._find_pair()
            if pair is None:
                raise ConnectorUnavailable(self._missing_reason(expired))
            occ_stamp, flow_stamp = pair
            _received, occ_msg = self._occupancy.pop(occ_stamp)
            _received, flow_msg = self._flow.pop(flow_stamp)
        try:
            occupancy = _retarget(occupancy_from_msg(occ_msg), "ros")
            flow = _retarget(flow_from_msg(flow_msg), "ros")
            validate_prediction_pair(occupancy, flow, self._tolerance_ns)
        except PredictionValidationError as exc:
            raise ConnectorUnavailable(f"ROS prediction failed validation: {exc}") from exc
        return occupancy, flow

    def _on_occupancy(self, message) -> None:
        self._store(self._occupancy, message)

    def _on_flow(self, message) -> None:
        self._store(self._flow, message)

    def _store(self, store: dict, message) -> None:
        stamp = (int(message.header.stamp.sec), int(message.header.stamp.nanosec))
        with self._lock:
            store[stamp] = (time.monotonic(), message)
            while len(store) > self._max_queue:
                oldest = next(iter(store))
                del store[oldest]

    def _expire(self) -> int:
        if self._timeout <= 0.0:
            return 0
        now = time.monotonic()
        dropped = 0
        for store in (self._occupancy, self._flow):
            stale = [
                stamp
                for stamp, (received, _message) in store.items()
                if now - received > self._timeout
            ]
            for stamp in stale:
                del store[stamp]
                dropped += 1
        return dropped

    def _find_pair(self):
        for occ_stamp in sorted(self._occupancy, key=lambda stamp: stamp_to_ns(*stamp)):
            occ_ns = stamp_to_ns(*occ_stamp)
            occ_frame = str(self._occupancy[occ_stamp][1].header.frame_id)
            candidates = []
            for flow_stamp, (_received, flow_msg) in self._flow.items():
                if str(flow_msg.header.frame_id) != occ_frame:
                    continue
                delta = abs(occ_ns - stamp_to_ns(*flow_stamp))
                if delta <= self._tolerance_ns:
                    candidates.append((delta, flow_stamp))
            if candidates:
                candidates.sort(key=lambda item: (item[0], stamp_to_ns(*item[1])))
                return occ_stamp, candidates[0][1]
        return None

    def _missing_reason(self, expired: int) -> str:
        occupancy_count = len(self._occupancy)
        flow_count = len(self._flow)
        if occupancy_count == 0 and flow_count == 0:
            reason = (
                "occupancy and flow messages are missing on "
                f"{self._occupancy_topic} and {self._flow_topic}"
            )
        elif occupancy_count and flow_count == 0:
            reason = (
                f"occupancy is buffered ({occupancy_count}) and the flow message is missing "
                f"on {self._flow_topic}"
            )
        elif flow_count and occupancy_count == 0:
            reason = (
                f"flow is buffered ({flow_count}) and the occupancy message is missing "
                f"on {self._occupancy_topic}"
            )
        else:
            reason = (
                "occupancy and flow are buffered but no pair shares a frame_id and "
                f"a timestamp within sync_tolerance_ns={self._tolerance_ns}"
            )
        if expired:
            reason += (
                f" Dropped {expired} unpaired message(s) older than "
                f"wait_timeout_sec={self._timeout}."
            )
        return (
            "ROS input expects OccupancyGrid3D and OccupancyFlow3D. " + reason
        )


def _retarget(prediction, source: str):
    data = prediction.__dict__.copy()
    data["source"] = source
    return type(prediction)(**data)
