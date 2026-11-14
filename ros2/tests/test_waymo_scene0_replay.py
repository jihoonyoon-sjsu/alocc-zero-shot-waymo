import json
import os
from pathlib import Path

import pytest

from alocc_ros.waymo_scene0_dataset import (
    CAMERA_NAMES,
    CSV_COLUMNS,
    PREDICTION_FILE,
    SceneFrame,
    WaymoSceneError,
    assert_prediction_frame_idx,
    camera_info_topic,
    decode_rgb_image,
    image_topic,
    load_calibration,
    load_frames,
    load_predictions,
    load_replay_scene,
    loop_wait_seconds,
    remaining_wait,
    occupancy_labels,
    playback_orders,
    ros_stamp,
    select_frames,
    validation_row,
    wait_seconds,
)

DATASET = Path(
    os.environ.get("WAYMO_SCENE0_DIR", Path.home() / "alocc-data" / "waymo_scene0")
).expanduser()


def _dataset_or_skip():
    if not (DATASET / "frames.json").is_file():
        pytest.skip("Waymo scene 0 dataset is not on this machine")
    return DATASET


@pytest.fixture(scope="session")
def scene(root):
    dataset = _dataset_or_skip()
    return load_replay_scene(dataset, root / "config" / "model.yaml")


def _write_archive(path: Path, missing_image: bool) -> None:
    frames = []
    for order in range(40):
        images = {}
        for name in CAMERA_NAMES:
            relative = f"images/{name}/{order:03d}.jpg"
            images[name] = relative
            destination = path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            omit = missing_image and order == 3 and name == "CAM_FRONT"
            if not omit:
                destination.write_bytes(b"placeholder")
        frames.append(
            {
                "order": order,
                "frame_idx": order * 5,
                "timestamp_us": 1_000_000 + order * 500_000,
                "images": images,
            }
        )
    (path / "frames.json").write_text(json.dumps(frames), encoding="utf-8")


def test_loads_forty_frames(scene):
    assert len(scene.frames) == 40
    assert [frame.order for frame in scene.frames] == list(range(40))


def test_frame_indices_step_by_five(scene):
    assert [frame.frame_idx for frame in scene.frames] == list(range(0, 200, 5))


def test_predictions_match_frame_indices(scene):
    assert scene.occupancy.shape[0] == 40
    assert scene.prediction_path.name == "zero_shot_scene0.npz"
    with pytest.raises(WaymoSceneError, match="do not match"):
        assert_prediction_frame_idx([0, 5, 10], [0, 5, 15])


def test_five_camera_views_exist(scene):
    assert [camera.name for camera in scene.cameras] == list(CAMERA_NAMES)
    for camera in scene.cameras:
        paths = [frame.image_paths[camera.name] for frame in scene.frames]
        assert len(paths) == 40
        assert all(path.is_file() for path in paths)
    front = next(camera for camera in scene.cameras if camera.name == "CAM_FRONT")
    side = next(camera for camera in scene.cameras if camera.name == "CAM_SIDE_LEFT")
    front_image = decode_rgb_image(scene.frames[0].image_paths["CAM_FRONT"])
    side_image = decode_rgb_image(scene.frames[0].image_paths["CAM_SIDE_LEFT"])
    assert (front_image[0], front_image[1]) == (front.height, front.width)
    assert (side_image[0], side_image[1]) == (side.height, side.width)
    assert image_topic("front") == "/alocc/waymo_scene0/camera/front/image_raw"
    assert camera_info_topic("side_left") == "/alocc/waymo_scene0/camera/side_left/camera_info"


def test_camera_calibration_intrinsics(scene):
    assert len(scene.cameras) == 5
    for camera in scene.cameras:
        assert camera.width == 1920
        assert camera.height in (1280, 886)
        assert len(camera.k) == 9
        assert camera.k[0] > 1000.0
        assert camera.k[8] == pytest.approx(1.0)
        assert camera.d is None
        assert camera.r is None
        assert camera.p is None
        assert camera.distortion_model is None


def test_timestamp_conversion():
    assert ros_stamp(1_522_688_014_970_187) == (1522688014, 970_187_000)
    with pytest.raises(WaymoSceneError):
        ros_stamp(-1)


def test_timestamps_increase(scene):
    stamps = [frame.timestamp_us for frame in scene.frames]
    assert stamps == sorted(stamps)
    assert all(later > earlier for earlier, later in zip(stamps, stamps[1:]))
    sec, nanosec = ros_stamp(stamps[0])
    assert sec * 1_000_000 + nanosec // 1_000 == stamps[0]


def test_eleven_messages_share_timestamp(scene):
    pytest.importorskip("sensor_msgs.msg")
    from alocc_ros.waymo_scene0_messages import build_camera_info, build_image, build_occupancy

    frame = scene.frames[0]
    sec, nanosec = ros_stamp(frame.timestamp_us)
    occupancy = build_occupancy(scene, frame)
    stamps = [(occupancy.header.stamp.sec, occupancy.header.stamp.nanosec)]
    for camera in scene.cameras:
        info = build_camera_info(camera, sec, nanosec)
        image = build_image(camera.name, sec, nanosec, 1, 1, b"\x00\x00\x00")
        stamps.append((info.header.stamp.sec, info.header.stamp.nanosec))
        stamps.append((image.header.stamp.sec, image.header.stamp.nanosec))
        assert list(info.d) == []
        assert list(info.r) == [0.0] * 9
        assert list(info.k) == pytest.approx(camera.k)
    assert len(stamps) == 11
    assert len(set(stamps)) == 1
    assert stamps[0] == (sec, nanosec)


def test_occupancy_shape_and_axis_order(scene):
    volume = scene.occupancy[0]
    assert volume.shape == (200, 200, 16)
    assert str(volume.dtype) == "uint8"
    labels = occupancy_labels(volume)
    assert len(labels) == 200 * 200 * 16
    assert labels[0] == int(volume[0, 0, 0])
    assert labels[1] == int(volume[1, 0, 0])
    assert labels[200] == int(volume[0, 1, 0])
    assert labels[200 * 200] == int(volume[0, 0, 1])


def test_ego_frame_id(scene):
    pytest.importorskip("alocc_interfaces.msg")
    from alocc_ros.waymo_scene0_messages import build_occupancy

    message = build_occupancy(scene, scene.frames[4])
    assert message.header.frame_id == "ego"
    assert message.geometry_confirmed is True
    assert message.flow_supplied is False
    assert message.timestamp_available is True
    assert message.is_mock is False
    assert (message.dim_x, message.dim_y, message.dim_z) == (200, 200, 16)


def test_semantic_class_mapping(scene, root):
    import yaml

    config = yaml.safe_load((root / "config" / "model.yaml").read_text())
    assert scene.class_names == tuple(config["confirmed_alocc"]["class_names"])
    assert scene.class_names[0] == "others"
    assert scene.class_names[17] == "free"
    assert len(scene.class_names) == 18
    assert int(scene.occupancy.min()) >= 0
    assert int(scene.occupancy.max()) <= 17


def test_replay_rate_uses_timestamp_deltas(scene):
    delta = scene.frames[1].timestamp_us - scene.frames[0].timestamp_us
    assert delta != 500_000
    assert wait_seconds(scene.frames[0].timestamp_us, scene.frames[1].timestamp_us, 1.0) == pytest.approx(
        delta / 1_000_000
    )
    assert wait_seconds(0, 500_000, 0.5) == pytest.approx(1.0)
    assert wait_seconds(0, 500_000, 2.0) == pytest.approx(0.25)
    assert remaining_wait(0.5, 0.2) == pytest.approx(0.3)
    assert remaining_wait(0.5, 0.8) == 0.0
    frames = scene.frames[:2]
    assert loop_wait_seconds(frames, 1.0) == pytest.approx(delta / 1_000_000)
    with pytest.raises(WaymoSceneError):
        wait_seconds(10, 10, 1.0)


def test_loop_playback_orders():
    assert playback_orders(0, 2, 2) == [0, 1, 2, 0, 1, 2]
    with pytest.raises(WaymoSceneError):
        playback_orders(3, 1, 1)


def test_frame_window_is_archive_order(scene):
    chosen = select_frames(scene, 0, 1)
    assert [frame.frame_idx for frame in chosen] == [0, 5]
    with pytest.raises(WaymoSceneError):
        select_frames(scene, 0, 195)


def test_missing_image(tmp_path):
    _write_archive(tmp_path, missing_image=True)
    with pytest.raises(WaymoSceneError, match="Missing image"):
        load_frames(tmp_path)


def test_missing_calibration(tmp_path):
    with pytest.raises(WaymoSceneError, match="Missing calibration.json"):
        load_calibration(tmp_path)


def test_missing_prediction(tmp_path):
    with pytest.raises(WaymoSceneError, match="Missing prediction file"):
        load_predictions(tmp_path, [], (200, 200, 16), 17)
    assert PREDICTION_FILE == "check/zero_shot_scene0.npz"


def test_validation_columns_and_replay_source(root):
    row = validation_row(0, 1522688014970187, 0, 5, 5, 1, "200x200x16", True, "ok")
    assert tuple(row) == CSV_COLUMNS
    node_source = (root / "src/alocc_ros/alocc_ros/waymo_scene0_replay_node.py").read_text()
    assert "OccupancyFlow3D" not in node_source
    assert "ego2global" not in node_source
    assert "/alocc/visualization/markers" not in node_source
    assert "time.time(" not in node_source
    dataset_source = (root / "src/alocc_ros/alocc_ros/waymo_scene0_dataset.py").read_text()
    assert "ego2global" in dataset_source
    assert "intentionally not read" in dataset_source


def test_scene_frame_does_not_store_ego2global():
    assert "ego2global" not in SceneFrame.__dataclass_fields__
