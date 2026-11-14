import pickle

import pytest

np = pytest.importorskip("numpy")

from alocc_ros.config_loader import load_yaml
from alocc_ros.connectors.base import ConnectorUnavailable
from alocc_ros.connectors.pickle_input import (
    PicklePredictionConnector,
    convert_saved_sample,
    summarize_pickle,
)
from alocc_ros.message_conversion import occupancy_to_msg
from alocc_ros.prediction_types import PredictionValidationError, SpatialMetadata
from alocc_ros.visualization_render import render_occupancy_slice_svg


def _array() -> np.ndarray:
    values = np.arange(24, dtype=np.uint8).reshape(4, 3, 2)
    return values


def _write(path, payload) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _connector(root, path, **overrides):
    model = load_yaml(root / "config" / "model.yaml")
    settings = {"path": str(path), "loop": False, "max_samples": 0}
    settings.update(overrides)
    model["pkl"] = settings
    return PicklePredictionConnector(model)


def test_nuscenes_samples_keep_occupancy_and_leave_flow_absent(root, tmp_path):
    array = _array()
    path = tmp_path / "nuscenes.pkl"
    _write(
        path,
        [
            {
                "index": 4,
                "pred_occupancy": array,
                "pred_flow": None,
                "pts_bbox": None,
                "iou": None,
                "pred_density": None,
            },
            {"index": 5, "pred_flow": None},
            {
                "index": 6,
                "pred_occupancy": array,
                "pred_flow": None,
                "pts_bbox": None,
                "iou": None,
                "pred_density": None,
            },
        ],
    )
    connector = _connector(root, path)
    first, first_flow = connector.load()
    second, second_flow = connector.load()
    skipped = connector.drain_skip_records()
    with pytest.raises(ConnectorUnavailable):
        connector.load()

    assert first_flow is None
    assert second_flow is None
    assert first.flow_supplied is False
    assert first.is_mock is False
    assert first.source == "pkl"
    assert first.timestamp_available is False
    assert first.timestamp_sec == 0
    assert first.spatial.geometry_confirmed is False
    assert (first.spatial.dim_x, first.spatial.dim_y, first.spatial.dim_z) == (4, 3, 2)
    assert first.semantic_labels[1 + 2 * 4 + 1 * 4 * 3] == int(array[1, 2, 1])
    assert first.class_names[int(array.max())] == f"unconfirmed_label_{int(array.max())}"
    assert "dataset=nuscenes" in first.sample_metadata
    assert "index=4" in first.sample_metadata
    assert "pred_flow=null" in first.sample_metadata
    assert second.sample_index == 2
    assert "index=6" in second.sample_metadata
    assert len(skipped) == 1
    assert skipped[0]["sample_index"] == "1"


def test_protocol2_batch_of_1001_keeps_every_sample_once(root, tmp_path):
    path = tmp_path / "batch.pkl"
    payload = [
        {
            "index": index,
            "pred_occupancy": np.full((2, 2, 2), index % 250, dtype=np.uint8),
            "pred_flow": None,
            "pred_density": None,
        }
        for index in range(1001)
    ]
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=2)
    summary = summarize_pickle(path)
    assert summary["sample_count"] == 1001
    assert summary["loaded_count"] == 1001
    assert summary["skipped_count"] == 0
    connector = _connector(root, path)
    seen = []
    for _ in range(1001):
        occupancy, flow = connector.load()
        assert flow is None
        seen.append(occupancy.sample_metadata)
    assert seen[0].count("index=0") == 1
    assert "index=1000" in seen[-1]
    assert len(seen) == len(set(seen))
    with pytest.raises(ConnectorUnavailable):
        connector.load()


def test_waymo_samples_have_scene_and_frame_without_a_flow_field(root, tmp_path):
    array = _array()
    path = tmp_path / "waymo.pkl"
    _write(
        path,
        [
            {"scene_idx": 3, "frame_idx": 10, "pred_occupancy": array},
            {"scene_idx": 3, "frame_idx": 15, "pred_occupancy": array},
        ],
    )
    connector = _connector(root, path, loop=True, max_samples=3)
    first, flow = connector.load()
    second, _ = connector.load()
    third, _ = connector.load()
    with pytest.raises(ConnectorUnavailable, match="max_samples"):
        connector.load()
    connector.close()

    assert flow is None
    assert "dataset=waymo" in first.sample_metadata
    assert "scene_idx=3" in first.sample_metadata
    assert "frame_idx=10" in first.sample_metadata
    assert "pred_flow=absent" in first.sample_metadata
    assert second.sample_index == 1
    assert third.sample_index == 0
    summary = summarize_pickle(path)
    assert summary["root_kind"] == "list"
    assert summary["dataset"] == "waymo"
    assert summary["sample_count"] == 2
    assert summary["loaded_count"] == 2
    assert summary["flow_absent_count"] == 2
    assert summary["flow_usable_count"] == 0
    assert summary["occupancy_shape"] == [4, 3, 2]
    assert summary["occupancy_dtype"] == "uint8"
    assert summary["shapes_consistent"] is True
    assert summary["timestamp_field_present"] is False


def test_unusable_flow_does_not_become_a_prediction(root, tmp_path):
    array = _array()
    two_component = np.zeros(array.shape + (2,), dtype=np.float32)
    path = tmp_path / "partial_flow.pkl"
    _write(
        path,
        [
            {
                "index": 0,
                "pred_occupancy": array,
                "pred_flow": two_component,
                "pred_density": None,
            }
        ],
    )
    occupancy, flow = _connector(root, path).load()
    assert flow is None
    assert occupancy.flow_supplied is False
    assert "pred_flow=present_but_not_usable" in occupancy.sample_metadata


def test_three_component_flow_is_kept_when_present():
    array = _array()
    flow_array = np.zeros(array.shape + (3,), dtype=np.float32)
    flow_array[1, 2, 1] = (0.25, -0.5, 1.5)
    occupancy, flow = convert_saved_sample(
        {"index": 1, "pred_occupancy": array, "pred_flow": flow_array, "pred_density": None},
        0,
    )
    assert occupancy.flow_supplied is True
    assert flow is not None
    offset = (1 + 2 * 4 + 1 * 4 * 3) * 3
    assert flow.flow_vectors[offset : offset + 3] == pytest.approx((0.25, -0.5, 1.5))
    assert flow.flow_units == "UNCONFIRMED_NOT_A_PHYSICAL_UNIT"


def test_placeholder_dimension_limit_still_rejects_80_and_pickle_limit_accepts_200():
    with pytest.raises(PredictionValidationError, match="exceeds the lightweight limit of 64"):
        SpatialMetadata(
            dim_x=80,
            dim_y=1,
            dim_z=1,
            resolution_x=0.5,
            resolution_y=0.5,
            resolution_z=0.5,
            origin_x=0.0,
            origin_y=0.0,
            origin_z=0.0,
            coordinate_convention="TEST_FRAME",
        ).validate()
    SpatialMetadata(
        dim_x=200,
        dim_y=200,
        dim_z=16,
        resolution_x=float("nan"),
        resolution_y=float("nan"),
        resolution_z=float("nan"),
        origin_x=float("nan"),
        origin_y=float("nan"),
        origin_z=float("nan"),
        coordinate_convention="PICKLE_AXIS0_FASTEST_AXIS1_THEN_AXIS2_GEOMETRY_UNCONFIRMED",
        geometry_confirmed=False,
        max_dimension=256,
    ).validate()


def test_real_slice_is_not_labelled_as_mock():
    array = _array()
    occupancy, _flow = convert_saved_sample(
        {"scene_idx": 0, "frame_idx": 0, "pred_occupancy": array},
        0,
    )
    svg = render_occupancy_slice_svg(occupancy)
    assert "REAL occupancy" in svg
    assert "geometry=unconfirmed_voxel_indices" in svg
    assert "MOCK occupancy" not in svg


def test_pickle_occupancy_round_trips_without_publishing_flow(ros_context, root, tmp_path):
    pytest.importorskip("alocc_interfaces.msg")
    from alocc_interfaces.msg import OccupancyFlow3D, OccupancyGrid3D

    array = _array()
    path = tmp_path / "roundtrip.pkl"
    _write(path, [{"index": 9, "pred_occupancy": array, "pred_flow": None, "pred_density": None}])
    occupancy, flow = _connector(root, path).load()
    assert flow is None
    message = occupancy_to_msg(occupancy, "alocc_placeholder")
    message.pipeline_publish_time_s = 12.5

    rclpy = ros_context
    publisher = rclpy.create_node("alocc_pkl_publisher")
    subscriber = rclpy.create_node("alocc_pkl_subscriber")
    received = {"occupancy": [], "flow": []}
    try:
        occ_pub = publisher.create_publisher(OccupancyGrid3D, "/alocc/test/pkl_occupancy", 10)
        subscriber.create_subscription(
            OccupancyGrid3D,
            "/alocc/test/pkl_occupancy",
            lambda msg: received["occupancy"].append(msg),
            10,
        )
        subscriber.create_subscription(
            OccupancyFlow3D,
            "/alocc/test/pkl_flow",
            lambda msg: received["flow"].append(msg),
            10,
        )

        def publish_and_check():
            occ_pub.publish(message)
            return bool(received["occupancy"])

        deadline = __import__("time").time() + 5.0
        while __import__("time").time() < deadline:
            rclpy.spin_once(publisher, timeout_sec=0.05)
            rclpy.spin_once(subscriber, timeout_sec=0.05)
            if publish_and_check():
                break
        assert received["occupancy"]
        assert received["flow"] == []
        got = received["occupancy"][0]
        assert list(got.semantic_labels) == list(occupancy.semantic_labels)
        assert (got.dim_x, got.dim_y, got.dim_z) == (4, 3, 2)
        assert got.flow_supplied is False
        assert got.is_mock is False
        assert got.geometry_confirmed is False
        assert got.timestamp_available is False
    finally:
        publisher.destroy_node()
        subscriber.destroy_node()


def test_confirmed_grid_uses_ego_geometry_and_official_classes(root):
    model = load_yaml(root / "config" / "model.yaml")
    array = np.full((200, 200, 16), 17, dtype=np.uint8)
    array[0, 0, 0] = 0
    array[100, 100, 5] = 4
    array[199, 199, 15] = 7
    array[1, 0, 0] = 18
    with pytest.raises(ConnectorUnavailable, match="outside"):
        convert_saved_sample(
            {"index": 1, "pred_occupancy": array, "pred_flow": None, "pred_density": None},
            0,
            model,
        )
    array[1, 0, 0] = 17
    occupancy, flow = convert_saved_sample(
        {"index": 1, "pred_occupancy": array, "pred_flow": None, "pred_density": None},
        0,
        model,
    )
    assert flow is None
    assert occupancy.flow_supplied is False
    assert occupancy.timestamp_available is False
    assert occupancy.timestamp_sec == 0
    assert occupancy.spatial.geometry_confirmed is True
    assert occupancy.class_names == tuple(model["confirmed_alocc"]["class_names"])
    assert occupancy.class_names[0] == "others"
    assert occupancy.class_names[4] == "car"
    assert occupancy.class_names[17] == "free"
    assert (occupancy.spatial.dim_x, occupancy.spatial.dim_y, occupancy.spatial.dim_z) == (200, 200, 16)
    assert occupancy.semantic_labels[0] == 0
    car_index = 100 + 100 * 200 + 5 * 200 * 200
    assert occupancy.semantic_labels[car_index] == 4
    assert occupancy.semantic_labels[-1] == 7
    assert "class_names=nuscenes_semantic_18" in occupancy.sample_metadata
    assert "timestamp=unavailable" in occupancy.sample_metadata
    assert "index=1" in occupancy.sample_metadata
    waymo, waymo_flow = convert_saved_sample(
        {"scene_idx": 0, "frame_idx": 3, "pred_occupancy": array},
        2,
        model,
    )
    assert waymo_flow is None
    assert waymo.spatial.geometry_confirmed is True
    assert waymo.class_names[7] == "pedestrian"
    assert "dataset=waymo" in waymo.sample_metadata
    assert "scene_idx=0" in waymo.sample_metadata
    assert "frame_idx=3" in waymo.sample_metadata
    assert "timestamp=unavailable" in waymo.sample_metadata
    assert waymo.semantic_labels[-1] == 7
