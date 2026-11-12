"""Waymo scene 0 archive, calibration, and verified occupancy loader.

The replay reads three files and does not modify them:

- ``frames.json`` supplies archive order, raw ``frame_idx``, and ``timestamp_us``.
- ``calibration.json`` supplies image size and the intrinsic matrix ``K``.
- ``check/zero_shot_scene0.npz`` supplies the 40 occupancy grids.

``timestamp_us`` is an absolute count of microseconds since the Unix epoch.
The scene values are about 1.522688e15, which is April 2018, not a relative
offset and not a frame counter. ``frames.json`` also contains ``ego2global``.
That matrix is intentionally not read. ``cam2ego`` is present in the
calibration file and is not applied, because this replay stays in the ego
frame and does not publish a transform tree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alocc_ros.config_loader import load_yaml, validate_model_config
from alocc_ros.occupancy_geometry import spatial_from_confirmed
from alocc_ros.prediction_types import OccupancyPrediction, SpatialMetadata

CAMERA_SPECS = (
    ("CAM_SIDE_LEFT", "side_left"),
    ("CAM_FRONT_LEFT", "front_left"),
    ("CAM_FRONT", "front"),
    ("CAM_FRONT_RIGHT", "front_right"),
    ("CAM_SIDE_RIGHT", "side_right"),
)
CAMERA_NAMES = tuple(name for name, _suffix in CAMERA_SPECS)
TOPIC_SUFFIX = {name: suffix for name, suffix in CAMERA_SPECS}
OCCUPANCY_TOPIC = "/alocc/waymo_scene0/occupancy"
EGO_FRAME_ID = "ego"
PREDICTION_FILE = "check/zero_shot_scene0.npz"
CSV_COLUMNS = (
    "frame_idx",
    "timestamp_us",
    "prediction_sample_index",
    "camera_images_published",
    "camera_info_published",
    "occupancy_published",
    "occupancy_shape",
    "timestamp_sync_valid",
    "status",
)


class WaymoSceneError(Exception):
    """The scene 0 archive or its replay settings are incomplete."""


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    topic_suffix: str
    width: int
    height: int
    k: tuple[float, ...]
    # None means the dataset does not provide the field.
    distortion_model: str | None
    d: tuple[float, ...] | None
    r: tuple[float, ...] | None
    p: tuple[float, ...] | None


@dataclass(frozen=True)
class SceneFrame:
    order: int
    frame_idx: int
    timestamp_us: int
    image_paths: dict[str, Path]


@dataclass(frozen=True)
class ReplayScene:
    frames: tuple[SceneFrame, ...]
    cameras: tuple[CameraCalibration, ...]
    occupancy: np.ndarray
    class_names: tuple[str, ...]
    spatial: SpatialMetadata
    prediction_path: Path


def image_topic(suffix: str) -> str:
    return f"/alocc/waymo_scene0/camera/{suffix}/image_raw"


def camera_info_topic(suffix: str) -> str:
    return f"/alocc/waymo_scene0/camera/{suffix}/camera_info"


def ros_stamp(timestamp_us: int) -> tuple[int, int]:
    """Convert dataset microseconds to ROS seconds and nanoseconds."""
    if isinstance(timestamp_us, bool) or not isinstance(timestamp_us, int):
        raise WaymoSceneError("timestamp_us must be an integer")
    if timestamp_us < 0:
        raise WaymoSceneError("timestamp_us must be non-negative")
    sec = timestamp_us // 1_000_000
    nanosec = (timestamp_us % 1_000_000) * 1_000
    if nanosec >= 1_000_000_000:
        raise WaymoSceneError("timestamp conversion produced an invalid nanosecond")
    return sec, nanosec


def remaining_wait(dataset_delay: float, elapsed: float) -> float:
    """Time left in a dataset interval after the frame has already been published."""
    if dataset_delay < 0.0:
        raise WaymoSceneError("dataset delay must be non-negative")
    return max(0.0, float(dataset_delay) - max(float(elapsed), 0.0))


def wait_seconds(previous_us: int, current_us: int, replay_rate: float) -> float:
    """Real-time delay between two dataset timestamps, scaled by replay_rate.

    ``replay_rate`` 1.0 keeps the archived interval. A smaller rate waits
    longer. The interval comes from the timestamps, not from a fixed 0.5 s.
    """
    if replay_rate <= 0.0:
        raise WaymoSceneError("replay_rate must be positive")
    delta = int(current_us) - int(previous_us)
    if delta <= 0:
        raise WaymoSceneError("replay timestamps must increase")
    return (delta / 1_000_000.0) / float(replay_rate)


def playback_orders(start_frame: int, end_frame: int, cycles: int) -> list[int]:
    """Archive-order indices visited by a finite number of playback cycles."""
    if cycles < 1:
        raise WaymoSceneError("cycles must be at least 1")
    if start_frame > end_frame:
        raise WaymoSceneError("start_frame must be <= end_frame")
    span = list(range(start_frame, end_frame + 1))
    return span * cycles


def select_frames(scene: ReplayScene, start_frame: int, end_frame: int) -> tuple[SceneFrame, ...]:
    """Select frames by archive order. ``end_frame`` 39 is the last sample, not raw frame 195."""
    last = len(scene.frames) - 1
    if start_frame < 0 or end_frame > last or start_frame > end_frame:
        raise WaymoSceneError(
            f"start_frame and end_frame must satisfy 0 <= start <= end <= {last}"
        )
    return scene.frames[start_frame : end_frame + 1]


def loop_wait_seconds(frames: tuple[SceneFrame, ...] | list[SceneFrame], replay_rate: float) -> float:
    """Delay used before a loop returns to the first selected frame."""
    if len(frames) >= 2:
        return wait_seconds(frames[-2].timestamp_us, frames[-1].timestamp_us, replay_rate)
    return 0.5 / replay_rate


def load_frames(dataset_path: Path) -> list[SceneFrame]:
    path = Path(dataset_path) / "frames.json"
    if not path.is_file():
        raise WaymoSceneError(f"Missing frames.json: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 40:
        raise WaymoSceneError("frames.json must contain 40 scene frames")
    frames: list[SceneFrame] = []
    previous = None
    for expected_order, item in enumerate(raw):
        if not isinstance(item, dict):
            raise WaymoSceneError(f"frame record {expected_order} is not an object")
        order = int(item["order"])
        frame_idx = int(item["frame_idx"])
        timestamp_us = int(item["timestamp_us"])
        if order != expected_order:
            raise WaymoSceneError(f"frame order {order} is not in archive position {expected_order}")
        if frame_idx != expected_order * 5:
            raise WaymoSceneError(f"frame_idx {frame_idx} does not match order {expected_order}")
        if previous is not None and timestamp_us <= previous:
            raise WaymoSceneError("frame timestamps are not strictly increasing")
        previous = timestamp_us
        images = item.get("images")
        if not isinstance(images, dict):
            raise WaymoSceneError(f"frame {frame_idx} has no image mapping")
        paths: dict[str, Path] = {}
        for name in CAMERA_NAMES:
            relative = images.get(name)
            if not isinstance(relative, str) or not relative:
                raise WaymoSceneError(f"frame {frame_idx} is missing image {name}")
            image_path = Path(dataset_path) / relative
            if not image_path.is_file():
                raise WaymoSceneError(f"Missing image for frame {frame_idx}: {image_path}")
            paths[name] = image_path
        frames.append(
            SceneFrame(
                order=order,
                frame_idx=frame_idx,
                timestamp_us=timestamp_us,
                image_paths=paths,
            )
        )
    return frames


def _matrix_3x3(value, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 3:
        raise WaymoSceneError(f"{name} must be a 3x3 matrix")
    flat: list[float] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            raise WaymoSceneError(f"{name} must be a 3x3 matrix")
        flat.extend(float(item) for item in row)
    return tuple(flat)


def _optional_floats(camera: dict, key: str) -> tuple[float, ...] | None:
    if key not in camera:
        return None
    value = camera[key]
    if not isinstance(value, list):
        raise WaymoSceneError(f"calibration {key} must be a list when it is present")
    return tuple(float(item) for item in value)


def load_calibration(dataset_path: Path) -> list[CameraCalibration]:
    path = Path(dataset_path) / "calibration.json"
    if not path.is_file():
        raise WaymoSceneError(f"Missing calibration.json: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise WaymoSceneError("calibration.json must be an object")
    order = raw.get("camera_order")
    cameras = raw.get("cameras")
    if list(order or []) != list(CAMERA_NAMES):
        raise WaymoSceneError("calibration camera_order does not match the five Waymo cameras")
    if not isinstance(cameras, list):
        raise WaymoSceneError("calibration cameras must be a list")
    by_name = {}
    for camera in cameras:
        if not isinstance(camera, dict) or "name" not in camera:
            raise WaymoSceneError("a calibration camera entry has no name")
        by_name[camera["name"]] = camera
    loaded = []
    for name, suffix in CAMERA_SPECS:
        camera = by_name.get(name)
        if camera is None:
            raise WaymoSceneError(f"calibration is missing {name}")
        width = int(camera["width"])
        height = int(camera["height"])
        if width <= 0 or height <= 0:
            raise WaymoSceneError(f"{name} has an invalid image size")
        loaded.append(
            CameraCalibration(
                name=name,
                topic_suffix=suffix,
                width=width,
                height=height,
                k=_matrix_3x3(camera.get("K"), f"{name} K"),
                distortion_model=camera.get("distortion_model"),
                d=_optional_floats(camera, "D"),
                r=_optional_floats(camera, "R"),
                p=_optional_floats(camera, "P"),
            )
        )
    return loaded


def assert_prediction_frame_idx(archive_frame_idx, prediction_frame_idx) -> None:
    archive = [int(value) for value in archive_frame_idx]
    prediction = [int(value) for value in prediction_frame_idx]
    if archive != prediction:
        raise WaymoSceneError("prediction frame_idx values do not match frames.json")


def load_predictions(dataset_path: Path, frames: list[SceneFrame], expected_dims, max_label: int) -> np.ndarray:
    path = Path(dataset_path) / PREDICTION_FILE
    if not path.is_file():
        raise WaymoSceneError(f"Missing prediction file: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "frame_idx" not in data.files or "pred_occupancy" not in data.files:
            raise WaymoSceneError("prediction npz is missing frame_idx or pred_occupancy")
        frame_idx = np.array(data["frame_idx"])
        occupancy = np.array(data["pred_occupancy"])
    assert_prediction_frame_idx([frame.frame_idx for frame in frames], frame_idx.tolist())
    expected_shape = (len(frames), int(expected_dims[0]), int(expected_dims[1]), int(expected_dims[2]))
    if occupancy.shape != expected_shape:
        raise WaymoSceneError(f"prediction shape {occupancy.shape} != {expected_shape}")
    if occupancy.dtype != np.uint8:
        raise WaymoSceneError(f"prediction dtype {occupancy.dtype} is not uint8")
    highest = int(occupancy.max())
    if highest > max_label:
        raise WaymoSceneError(f"prediction label {highest} is outside the confirmed class map")
    return occupancy


def load_replay_scene(dataset_path: str | Path, model_config_path: str | Path) -> ReplayScene:
    dataset = Path(dataset_path).expanduser()
    config = load_yaml(Path(model_config_path).expanduser())
    validate_model_config(config)
    section = config["confirmed_alocc"]
    spatial = spatial_from_confirmed(section)
    class_names = tuple(section["class_names"])
    frames = load_frames(dataset)
    cameras = load_calibration(dataset)
    occupancy = load_predictions(
        dataset,
        frames,
        (spatial.dim_x, spatial.dim_y, spatial.dim_z),
        len(class_names) - 1,
    )
    return ReplayScene(
        frames=tuple(frames),
        cameras=tuple(cameras),
        occupancy=occupancy,
        class_names=class_names,
        spatial=spatial,
        prediction_path=dataset / PREDICTION_FILE,
    )


def occupancy_labels(volume: np.ndarray) -> tuple[int, ...]:
    """Flatten one grid with x fastest, then y, then z. The array is not transposed."""
    flat = np.asarray(volume).ravel(order="F")
    return tuple(int(label) for label in flat.tolist())


def frame_metadata(frame: SceneFrame) -> str:
    return (
        "dataset=waymo;scene_idx=0;"
        f"order={frame.order};frame_idx={frame.frame_idx};"
        f"timestamp_us={frame.timestamp_us};"
        "occupancy_source=check/zero_shot_scene0.npz;flow_supplied=false"
    )


def occupancy_prediction(scene: ReplayScene, frame: SceneFrame) -> OccupancyPrediction:
    sec, nanosec = ros_stamp(frame.timestamp_us)
    prediction = OccupancyPrediction(
        spatial=scene.spatial,
        class_names=scene.class_names,
        semantic_labels=occupancy_labels(scene.occupancy[frame.order]),
        timestamp_sec=sec,
        timestamp_nanosec=nanosec,
        is_mock=False,
        source="pkl",
        timestamp_available=True,
        sample_index=frame.order,
        sample_metadata=frame_metadata(frame),
        flow_supplied=False,
    )
    prediction.validate()
    return prediction


def decode_rgb_image(path: Path) -> tuple[int, int, bytes]:
    """Decode a JPEG to packed RGB bytes. Encoding for the ROS image is rgb8."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise WaymoSceneError("Pillow is required to decode Waymo JPEG images") from exc
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        data = rgb.tobytes()
    if len(data) != width * height * 3:
        raise WaymoSceneError(f"decoded image size is inconsistent: {path}")
    return height, width, data


def parse_metadata(text: str) -> dict[str, str]:
    fields = {}
    for item in text.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def validation_row(
    frame_idx: int,
    timestamp_us: int,
    prediction_sample_index: int,
    camera_images_published: int,
    camera_info_published: int,
    occupancy_published: int,
    occupancy_shape: str,
    timestamp_sync_valid: bool,
    status: str,
) -> dict[str, str]:
    return {
        "frame_idx": str(frame_idx),
        "timestamp_us": str(timestamp_us),
        "prediction_sample_index": str(prediction_sample_index),
        "camera_images_published": str(camera_images_published),
        "camera_info_published": str(camera_info_published),
        "occupancy_published": str(occupancy_published),
        "occupancy_shape": occupancy_shape,
        "timestamp_sync_valid": "true" if timestamp_sync_valid else "false",
        "status": status,
    }
