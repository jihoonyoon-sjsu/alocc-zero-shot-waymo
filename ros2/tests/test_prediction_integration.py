"""Integration checks for the file, Python, and ROS prediction contracts.

The grids below are synthetic. They are not ALOcc, nuScenes, or Waymo output.
"""

import itertools
import json
import math
import time

import pytest

from alocc_ros.config_loader import ConfigError, load_yaml, validate_mode
from alocc_ros.connectors.base import ConnectorUnavailable
from alocc_ros.connectors.file_input import FilePredictionConnector
from alocc_ros.connectors.python_inference import PredictionSample, PythonInferenceConnector
from alocc_ros.connectors.ros_subscription import RosSubscriptionConnector
from alocc_ros.prediction_types import (
    FlowPrediction,
    OccupancyPrediction,
    PredictionValidationError,
    SpatialMetadata,
    validate_prediction_pair,
)

CLASSES = (
    "synthetic_unlabeled",
    "synthetic_free",
    "synthetic_vehicle",
    "synthetic_pedestrian",
)
DIMS = (4, 3, 2)
STAMP_SEC = 1_700_000_000
_NODE_IDS = itertools.count(1)


def _index(x: int, y: int, z: int = 0) -> int:
    return x + y * DIMS[0] + z * DIMS[0] * DIMS[1]


def _spatial() -> SpatialMetadata:
    return SpatialMetadata(
        dim_x=DIMS[0],
        dim_y=DIMS[1],
        dim_z=DIMS[2],
        resolution_x=0.4,
        resolution_y=0.4,
        resolution_z=0.4,
        origin_x=-2.0,
        origin_y=-1.2,
        origin_z=-0.4,
        coordinate_convention="SYNTHETIC_TEST_XYZ",
    )


def make_pair(
    source: str,
    timestamp_sec: int,
    timestamp_nanosec: int,
    vehicle: tuple[int, int, int],
    pedestrian: tuple[int, int, int] | None = None,
    velocity: tuple[float, float, float] = (0.4, 0.0, 0.0),
) -> tuple[OccupancyPrediction, FlowPrediction]:
    labels = [1] * (DIMS[0] * DIMS[1] * DIMS[2])
    flow = [0.0] * (DIMS[0] * DIMS[1] * DIMS[2] * 3)
    vehicle_index = _index(*vehicle)
    labels[vehicle_index] = 2
    base = vehicle_index * 3
    flow[base : base + 3] = list(velocity)
    if pedestrian is not None:
        pedestrian_index = _index(*pedestrian)
        labels[pedestrian_index] = 3
        pedestrian_base = pedestrian_index * 3
        flow[pedestrian_base : pedestrian_base + 3] = [0.0, 0.1, 0.0]
    occupancy = OccupancyPrediction(
        spatial=_spatial(),
        class_names=CLASSES,
        semantic_labels=tuple(labels),
        timestamp_sec=timestamp_sec,
        timestamp_nanosec=timestamp_nanosec,
        is_mock=False,
        source=source,
    )
    flow_prediction = FlowPrediction(
        spatial=_spatial(),
        flow_vectors=tuple(flow),
        timestamp_sec=timestamp_sec,
        timestamp_nanosec=timestamp_nanosec,
        is_mock=False,
        source=source,
        flow_units="SYNTHETIC_METERS_PER_SECOND",
    )
    validate_prediction_pair(occupancy, flow_prediction)
    return occupancy, flow_prediction


def _payload(occupancy: OccupancyPrediction, flow: FlowPrediction, sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "is_mock": False,
        "timestamp_sec": occupancy.timestamp_sec,
        "timestamp_nanosec": occupancy.timestamp_nanosec,
        "spatial": dict(occupancy.spatial.__dict__),
        "class_names": list(occupancy.class_names),
        "semantic_labels": list(occupancy.semantic_labels),
        "flow_vectors": list(flow.flow_vectors),
        "flow_units": flow.flow_units,
        "label_layout": occupancy.label_layout,
        "vector_layout": flow.vector_layout,
    }


def _write_frame(directory, name, vehicle, nanosec, pedestrian=None, sample_id="synthetic"):
    occupancy, flow = make_pair("file", STAMP_SEC, nanosec, vehicle, pedestrian)
    payload = _payload(occupancy, flow, sample_id)
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _spin_until(rclpy, nodes, predicate, timeout=5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for node in nodes:
            rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
    return False


class _RosHarness:
    def __init__(self, rclpy, root, **ros_input):
        pytest.importorskip("alocc_interfaces.msg")
        from alocc_interfaces.msg import OccupancyFlow3D, OccupancyGrid3D
        from alocc_ros.message_conversion import flow_to_msg, occupancy_to_msg

        self._occupancy_to_msg = occupancy_to_msg
        self._flow_to_msg = flow_to_msg
        self.rclpy = rclpy
        number = next(_NODE_IDS)
        model = load_yaml(root / "config" / "model.yaml")
        ros_config = load_yaml(root / "config" / "ros2.yaml")
        model["ros_input"] = {
            "occupancy_topic": f"/alocc/test/integ_occ_{number}",
            "flow_topic": f"/alocc/test/integ_flow_{number}",
            "wait_timeout_sec": 5.0,
            "sync_tolerance_ns": 0,
            "max_queue": 16,
        }
        model["ros_input"].update(ros_input)
        self.model = model
        self.source = rclpy.create_node(f"alocc_integ_src_{number}")
        self.listener = rclpy.create_node(f"alocc_integ_in_{number}")
        self.connector = RosSubscriptionConnector(model, ros_config, self.listener)
        self.occ_pub = self.source.create_publisher(
            OccupancyGrid3D, model["ros_input"]["occupancy_topic"], 10
        )
        self.flow_pub = self.source.create_publisher(
            OccupancyFlow3D, model["ros_input"]["flow_topic"], 10
        )

    def publish(self, occupancy=None, flow=None, occ_frame="synthetic_frame", flow_frame="synthetic_frame"):
        if occupancy is not None:
            self.occ_pub.publish(self._occupancy_to_msg(occupancy, occ_frame))
        if flow is not None:
            self.flow_pub.publish(self._flow_to_msg(flow, flow_frame))

    def spin_until(self, predicate, timeout=5.0) -> bool:
        return _spin_until(self.rclpy, [self.source, self.listener], predicate, timeout)

    def close(self) -> None:
        self.source.destroy_node()
        self.listener.destroy_node()


def test_shape_errors_name_occupancy_and_flow_dimensions():
    occupancy, flow = make_pair("mock", STAMP_SEC, 0, (1, 1, 0))
    short_labels = OccupancyPrediction(
        **{**occupancy.__dict__, "semantic_labels": occupancy.semantic_labels[:-1]}
    )
    with pytest.raises(PredictionValidationError, match=r"shape \(4, 3, 2\)"):
        short_labels.validate()
    outside = OccupancyPrediction(
        **{**occupancy.__dict__, "semantic_labels": (9,) + occupancy.semantic_labels[1:]}
    )
    with pytest.raises(PredictionValidationError, match="outside class_names"):
        outside.validate()
    short_flow = FlowPrediction(**{**flow.__dict__, "flow_vectors": flow.flow_vectors[:-1]})
    with pytest.raises(PredictionValidationError, match=r"shape \(4, 3, 2, 3\)"):
        short_flow.validate()
    huge = SpatialMetadata(**{**occupancy.spatial.__dict__, "dim_x": 80})
    with pytest.raises(PredictionValidationError, match="exceeds the lightweight limit of 64"):
        huge.validate()


def test_pair_rejects_different_timestamps():
    occupancy, flow = make_pair("mock", STAMP_SEC, 0, (1, 1, 0))
    shifted = FlowPrediction(**{**flow.__dict__, "timestamp_nanosec": 5})
    with pytest.raises(PredictionValidationError, match="differ by 5 ns"):
        validate_prediction_pair(occupancy, shifted)
    validate_prediction_pair(occupancy, shifted, timestamp_tolerance_ns=5)


def test_file_directory_plays_frames_in_filename_order(root, tmp_path):
    model = load_yaml(root / "config" / "model.yaml")
    directory = tmp_path / "predictions"
    directory.mkdir()
    _write_frame(directory, "0003.json", (3, 1, 0), 200_000_000, (0, 2, 0), "synthetic-0003")
    _write_frame(directory, "0001.json", (1, 1, 0), 0, sample_id="synthetic-0001")
    _write_frame(directory, "0002.json", (2, 1, 0), 100_000_000, sample_id="synthetic-0002")
    model["file"] = {"path": str(directory), "loop": False}
    validate_mode("file", model)
    connector = FilePredictionConnector(model)

    first, first_flow = connector.load()
    second, _second_flow = connector.load()
    third, third_flow = connector.load()

    assert [item.timestamp_nanosec for item in (first, second, third)] == [
        0,
        100_000_000,
        200_000_000,
    ]
    assert first.semantic_labels[_index(1, 1, 0)] == 2
    assert second.semantic_labels[_index(2, 1, 0)] == 2
    assert third.semantic_labels[_index(3, 1, 0)] == 2
    assert third.semantic_labels[_index(0, 2, 0)] == 3
    assert first_flow.flow_vectors[_index(1, 1, 0) * 3] == pytest.approx(0.4)
    assert third_flow.flow_vectors[_index(0, 2, 0) * 3 + 1] == pytest.approx(0.1)
    assert first.source == "file" and first.is_mock is False
    assert first.spatial.dim_x == 4
    with pytest.raises(ConnectorUnavailable, match="No more prediction files"):
        connector.load()


def test_single_prediction_file_is_reread_when_looping(root, tmp_path):
    model = load_yaml(root / "config" / "model.yaml")
    path = _write_frame(tmp_path, "prediction.json", (1, 1, 0), 0)
    model["file"] = {"path": str(path)}
    connector = FilePredictionConnector(model)
    first, _flow = connector.load()
    second, _flow = connector.load()
    assert first.semantic_labels == second.semantic_labels
    assert first.timestamp_sec == second.timestamp_sec == STAMP_SEC


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ("labels", r"shape \(4, 3, 2\)"),
        ("flow", r"shape \(4, 3, 2, 3\)"),
        ("class_index", "outside class_names"),
        ("dimension", "exceeds the lightweight limit of 64"),
    ],
)
def test_file_connector_rejects_invalid_prediction_shapes(root, tmp_path, mutate, message):
    model = load_yaml(root / "config" / "model.yaml")
    occupancy, flow = make_pair("file", STAMP_SEC, 0, (1, 1, 0))
    payload = _payload(occupancy, flow, "synthetic-bad")
    if mutate == "labels":
        payload["semantic_labels"] = payload["semantic_labels"][:-1]
    elif mutate == "flow":
        payload["flow_vectors"] = payload["flow_vectors"][:-1]
    elif mutate == "class_index":
        payload["semantic_labels"][0] = 9
    else:
        payload["spatial"]["dim_x"] = 80
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    model["file"] = {"path": str(path), "loop": False}
    with pytest.raises(ConnectorUnavailable, match=message):
        FilePredictionConnector(model).load()


def test_file_directory_skips_a_bad_frame_and_continues(root, tmp_path):
    model = load_yaml(root / "config" / "model.yaml")
    directory = tmp_path / "mixed"
    directory.mkdir()
    _write_frame(directory, "0001.json", (1, 1, 0), 0)
    bad_path = _write_frame(directory, "0002.json", (2, 1, 0), 100_000_000)
    payload = json.loads(bad_path.read_text(encoding="utf-8"))
    payload["flow_vectors"] = payload["flow_vectors"][:-1]
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    _write_frame(directory, "0003.json", (3, 1, 0), 200_000_000)
    model["file"] = {"path": str(directory), "loop": False}
    connector = FilePredictionConnector(model)
    first, _flow = connector.load()
    with pytest.raises(ConnectorUnavailable, match="flow_vectors"):
        connector.load()
    third, _flow = connector.load()
    assert first.timestamp_nanosec == 0
    assert third.timestamp_nanosec == 200_000_000


def test_file_mode_rejects_an_empty_directory(root, tmp_path):
    model = load_yaml(root / "config" / "model.yaml")
    empty = tmp_path / "empty"
    empty.mkdir()
    model["file"] = {"path": str(empty)}
    with pytest.raises(ConfigError, match="No prediction files"):
        validate_mode("file", model)


def test_python_one_argument_function_still_works_with_a_sample(root):
    model = load_yaml(root / "config" / "model.yaml")
    model["python"] = {
        "callable": "synthetic_module:predict",
        "sample": {"sample_id": "not-required", "metadata": {"scene_token": "synthetic"}},
    }
    validate_mode("python", model)

    def predict(model_config):
        return make_pair("python", STAMP_SEC, 0, (1, 1, 0))

    occupancy, flow = PythonInferenceConnector(model, predict_fn=predict).load()
    assert occupancy.source == "python"
    assert flow.source == "python"
    assert occupancy.semantic_labels[_index(1, 1, 0)] == 2


def test_python_function_receives_a_sample_sequence(root, tmp_path):
    model = load_yaml(root / "config" / "model.yaml")
    directory = tmp_path / "samples"
    directory.mkdir()
    samples = [
        ("0001.json", "synthetic-scene-7-frame-0", 0, (1, 1, 0), None),
        ("0002.json", "synthetic-scene-7-frame-1", 100_000_000, (2, 1, 0), (0, 2, 0)),
    ]
    for name, sample_id, nanosec, _vehicle, _pedestrian in samples:
        payload = {
            "sample_id": sample_id,
            "timestamp_sec": STAMP_SEC,
            "timestamp_nanosec": nanosec,
            "metadata": {"scene_token": "synthetic-scene-7"},
        }
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")
    model["python"] = {
        "callable": "synthetic_module:predict",
        "sample_path": str(directory),
        "loop": False,
        "sample": {
            "metadata": {"sensor": "synthetic-lidar"},
            "payload": {"split": "test"},
        },
    }
    validate_mode("python", model)
    seen = []

    def predict(model_config, sample: PredictionSample):
        assert model_config["occupancy"]["dim_x"] == 8
        assert isinstance(sample, PredictionSample)
        assert sample.metadata["sensor"] == "synthetic-lidar"
        assert sample.metadata["scene_token"] == "synthetic-scene-7"
        assert sample.payload["split"] == "test"
        seen.append(sample.sample_id)
        vehicle = (1, 1, 0) if sample.timestamp_nanosec == 0 else (2, 1, 0)
        pedestrian = None if sample.timestamp_nanosec == 0 else (0, 2, 0)
        return make_pair(
            "python",
            sample.timestamp_sec,
            sample.timestamp_nanosec,
            vehicle,
            pedestrian,
        )

    connector = PythonInferenceConnector(model, predict_fn=predict)
    first, _flow = connector.load()
    second, _flow = connector.load()
    assert seen == ["synthetic-scene-7-frame-0", "synthetic-scene-7-frame-1"]
    assert first.timestamp_nanosec == 0
    assert second.semantic_labels[_index(0, 2, 0)] == 3
    with pytest.raises(ConnectorUnavailable, match="No more Python samples"):
        connector.load()


def test_python_result_with_bad_labels_is_rejected(root):
    model = load_yaml(root / "config" / "model.yaml")
    occupancy, flow = make_pair("python", STAMP_SEC, 0, (1, 1, 0))

    def predict(model_config, sample):
        broken = OccupancyPrediction(
            **{**occupancy.__dict__, "semantic_labels": occupancy.semantic_labels[:-1]}
        )
        return broken, flow

    model["python"] = {
        "callable": "synthetic_module:predict",
        "sample": {"sample_id": "synthetic-bad-labels"},
    }
    with pytest.raises(ConnectorUnavailable, match="synthetic-bad-labels"):
        PythonInferenceConnector(model, predict_fn=predict).load()


def test_ros_pairs_matching_timestamps(ros_context, root):
    harness = _RosHarness(ros_context, root)
    occupancy, flow = make_pair("ros", STAMP_SEC, 0, (1, 1, 0))
    try:

        def publish():
            harness.publish(occupancy, flow)
            return harness.connector.ready()

        assert harness.spin_until(publish)
        loaded, loaded_flow = harness.connector.load()
        assert loaded.source == "ros"
        assert loaded_flow.source == "ros"
        assert loaded.timestamp_sec == STAMP_SEC
        assert loaded.semantic_labels == occupancy.semantic_labels
        assert len(loaded_flow.flow_vectors) == len(flow.flow_vectors)
        assert all(
            math.isclose(actual, expected, abs_tol=1e-6)
            for actual, expected in zip(loaded_flow.flow_vectors, flow.flow_vectors)
        )
        assert harness.connector.buffered_counts() == (0, 0)
    finally:
        harness.close()


def test_ros_keeps_occupancy_until_flow_arrives(ros_context, root):
    harness = _RosHarness(ros_context, root)
    occupancy, flow = make_pair("ros", STAMP_SEC, 0, (2, 1, 0))
    try:

        def publish_occupancy():
            harness.publish(occupancy=occupancy)
            return harness.connector.buffered_counts() == (1, 0)

        assert harness.spin_until(publish_occupancy)
        with pytest.raises(ConnectorUnavailable, match="flow message is missing"):
            harness.connector.load()
        assert harness.connector.buffered_counts() == (1, 0)

        def publish_flow():
            harness.publish(flow=flow)
            return harness.connector.ready()

        assert harness.spin_until(publish_flow)
        loaded, _loaded_flow = harness.connector.load()
        assert loaded.semantic_labels[_index(2, 1, 0)] == 2
    finally:
        harness.close()


def test_ros_does_not_pair_timestamps_outside_tolerance(ros_context, root):
    harness = _RosHarness(ros_context, root, sync_tolerance_ns=50_000_000)
    occupancy, flow = make_pair("ros", STAMP_SEC, 0, (1, 1, 0))
    late_flow = FlowPrediction(**{**flow.__dict__, "timestamp_nanosec": 80_000_000})
    exact_flow = flow
    try:

        def publish_unsynced():
            harness.publish(occupancy, late_flow)
            return harness.connector.buffered_counts() == (1, 1)

        assert harness.spin_until(publish_unsynced)
        with pytest.raises(ConnectorUnavailable, match="sync_tolerance_ns"):
            harness.connector.load()
        assert harness.connector.buffered_counts() == (1, 1)

        def publish_match():
            harness.publish(flow=exact_flow)
            return harness.connector.ready()

        assert harness.spin_until(publish_match)
        loaded, loaded_flow = harness.connector.load()
        assert loaded.timestamp_nanosec == 0
        assert loaded_flow.timestamp_nanosec == 0
        assert harness.connector.buffered_counts() == (0, 1)
    finally:
        harness.close()


def test_ros_pairs_timestamps_within_tolerance(ros_context, root):
    harness = _RosHarness(ros_context, root, sync_tolerance_ns=50_000_000)
    occupancy, flow = make_pair("ros", STAMP_SEC, 0, (1, 1, 0))
    near_flow = FlowPrediction(**{**flow.__dict__, "timestamp_nanosec": 40_000_000})
    try:

        def publish():
            harness.publish(occupancy, near_flow)
            return harness.connector.ready()

        assert harness.spin_until(publish)
        loaded, loaded_flow = harness.connector.load()
        assert loaded.timestamp_nanosec == 0
        assert loaded_flow.timestamp_nanosec == 40_000_000
    finally:
        harness.close()


def test_ros_requires_the_same_frame_id(ros_context, root):
    harness = _RosHarness(ros_context, root)
    occupancy, flow = make_pair("ros", STAMP_SEC, 0, (1, 1, 0))
    try:

        def publish_mismatch():
            harness.publish(occupancy, flow, occ_frame="synthetic_a", flow_frame="synthetic_b")
            return harness.connector.buffered_counts() == (1, 1)

        assert harness.spin_until(publish_mismatch)
        with pytest.raises(ConnectorUnavailable, match="frame_id"):
            harness.connector.load()

        def publish_match():
            harness.publish(flow=flow, flow_frame="synthetic_a")
            return harness.connector.ready()

        assert harness.spin_until(publish_match)
        loaded, _flow = harness.connector.load()
        assert loaded.source == "ros"
    finally:
        harness.close()


def test_ros_drops_an_invalid_shape_and_accepts_the_next_pair(ros_context, root):
    harness = _RosHarness(ros_context, root)
    occupancy, flow = make_pair("ros", STAMP_SEC, 0, (1, 1, 0))
    follow_occ, follow_flow = make_pair("ros", STAMP_SEC, 100_000_000, (2, 1, 0))
    try:
        bad = harness._occupancy_to_msg(occupancy, "synthetic_frame")
        bad.semantic_labels = [0]
        sent = {"bad": False, "good": False}

        def publish_bad():
            if not sent["bad"]:
                harness.occ_pub.publish(bad)
                harness.publish(flow=flow)
                sent["bad"] = True
            return harness.connector.ready()

        assert harness.spin_until(publish_bad)
        with pytest.raises(ConnectorUnavailable, match=r"shape \(4, 3, 2\)"):
            harness.connector.load()
        assert harness.connector.buffered_counts() == (0, 0)

        def publish_good():
            if not sent["good"]:
                harness.publish(follow_occ, follow_flow)
                sent["good"] = True
            return harness.connector.ready()

        assert harness.spin_until(publish_good)
        loaded, _loaded_flow = harness.connector.load()
        assert loaded.timestamp_nanosec == 100_000_000
        assert loaded.semantic_labels[_index(2, 1, 0)] == 2
    finally:
        harness.close()


def test_ros_drops_unpaired_messages_after_the_wait_timeout(ros_context, root):
    harness = _RosHarness(ros_context, root, wait_timeout_sec=0.2)
    occupancy, _flow = make_pair("ros", STAMP_SEC, 0, (1, 1, 0))
    try:

        def publish_occupancy():
            harness.publish(occupancy=occupancy)
            return harness.connector.buffered_counts() == (1, 0)

        assert harness.spin_until(publish_occupancy)
        time.sleep(0.45)
        with pytest.raises(ConnectorUnavailable, match="Dropped 1 unpaired"):
            harness.connector.load()
        assert harness.connector.buffered_counts() == (0, 0)
    finally:
        harness.close()


def test_ros_queue_keeps_the_newest_unmatched_occupancy(ros_context, root):
    harness = _RosHarness(ros_context, root, max_queue=1)
    older, _older_flow = make_pair("ros", STAMP_SEC, 0, (1, 1, 0))
    newer, newer_flow = make_pair("ros", STAMP_SEC, 100_000_000, (2, 1, 0))
    try:

        def publish_older():
            harness.publish(occupancy=older)
            return harness.connector.buffered_stamps()[0] == ((STAMP_SEC, 0),)

        assert harness.spin_until(publish_older)

        def publish_newer():
            harness.publish(occupancy=newer)
            return harness.connector.buffered_stamps()[0] == ((STAMP_SEC, 100_000_000),)

        assert harness.spin_until(publish_newer)

        def publish_old_flow():
            harness.publish(flow=_older_flow)
            return harness.connector.buffered_counts()[1] == 1

        assert harness.spin_until(publish_old_flow)
        with pytest.raises(ConnectorUnavailable, match="sync_tolerance_ns"):
            harness.connector.load()

        def publish_new_flow():
            harness.publish(flow=newer_flow)
            return harness.connector.ready()

        assert harness.spin_until(publish_new_flow)
        loaded, _flow = harness.connector.load()
        assert loaded.timestamp_nanosec == 100_000_000
    finally:
        harness.close()
