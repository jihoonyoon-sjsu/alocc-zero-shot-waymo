"""Load and validate framework YAML configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from alocc_ros.occupancy_geometry import spatial_from_confirmed
from alocc_ros.prediction_types import PredictionValidationError, SpatialMetadata


class ConfigError(PredictionValidationError):
    """Raised when a framework configuration cannot be used."""


REQUIRED_OCCUPANCY_KEYS = (
    "dim_x",
    "dim_y",
    "dim_z",
    "resolution_x",
    "resolution_y",
    "resolution_z",
    "origin_x",
    "origin_y",
    "origin_z",
    "coordinate_convention",
    "class_names",
)


def load_yaml(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"configuration file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"configuration root must be a mapping: {file_path}")
    return data


def validate_model_config(config: dict) -> None:
    occupancy = config.get("occupancy")
    flow = config.get("flow")
    if not isinstance(occupancy, dict) or not isinstance(flow, dict):
        raise ConfigError("model config needs occupancy and flow mappings")
    missing = [key for key in REQUIRED_OCCUPANCY_KEYS if key not in occupancy]
    if missing:
        raise ConfigError("occupancy config missing keys: " + ", ".join(missing))
    for key in ("flow_units", "vector_layout"):
        if key not in flow:
            raise ConfigError(f"flow config missing {key}")
    spatial_from_config(config, for_flow=False).validate()
    spatial_from_config(config, for_flow=True).validate()
    names = occupancy["class_names"]
    if not isinstance(names, list) or not names:
        raise ConfigError("occupancy.class_names must be a non-empty list")
    validate_confirmed_alocc(config)


def validate_confirmed_alocc(config: dict) -> None:
    """Check the shared ALOcc occupancy contract and the grid bounds.

    The mock occupancy block stays separate. This function reads
    confirmed_alocc and rejects a grid whose bounds do not match
    origin + dimensions * voxel size.
    """
    section = config.get("confirmed_alocc")
    if not isinstance(section, dict):
        raise ConfigError("model config needs a confirmed_alocc mapping")
    required = (
        "dim_x",
        "dim_y",
        "dim_z",
        "resolution_x",
        "resolution_y",
        "resolution_z",
        "origin_x",
        "origin_y",
        "origin_z",
        "bound_x_max",
        "bound_y_max",
        "bound_z_max",
        "coordinate_frame",
        "axis_order",
        "axis_directions",
        "coordinate_convention",
        "class_names",
    )
    missing = [key for key in required if key not in section]
    if missing:
        raise ConfigError("confirmed_alocc missing keys: " + ", ".join(missing))
    names = section["class_names"]
    if not isinstance(names, list) or len(names) != 18:
        raise ConfigError("confirmed_alocc.class_names must list exactly 18 classes")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ConfigError("confirmed_alocc class names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ConfigError("confirmed_alocc class names must be unique")
    if names[0] != "others" or names[17] != "free":
        raise ConfigError("confirmed class id 0 must be others and id 17 must be free")
    if list(section["axis_order"]) != ["x", "y", "z"]:
        raise ConfigError("confirmed_alocc.axis_order must be [x, y, z]")
    directions = section["axis_directions"]
    if not isinstance(directions, dict):
        raise ConfigError("confirmed_alocc.axis_directions must be a mapping")
    expected_directions = {"x": "forward", "y": "left", "z": "up"}
    for axis, direction in expected_directions.items():
        if str(directions.get(axis, "")) != direction:
            raise ConfigError(f"confirmed_alocc axis {axis} must point {direction}")
    if str(section["coordinate_frame"]) != "ego":
        raise ConfigError("confirmed_alocc.coordinate_frame must be ego")
    if str(section["coordinate_convention"]) != "ego_x_forward_y_left_z_up":
        raise ConfigError("confirmed_alocc.coordinate_convention must describe the ego frame")
    spatial = spatial_from_confirmed(section)
    spatial.validate()
    for axis in ("x", "y", "z"):
        origin = float(section[f"origin_{axis}"])
        resolution = float(section[f"resolution_{axis}"])
        dimension = int(section[f"dim_{axis}"])
        bound = float(section[f"bound_{axis}_max"])
        expected = origin + dimension * resolution
        if abs(expected - bound) > 1e-6:
            raise ConfigError(
                f"confirmed_alocc {axis} bound {bound} does not match "
                f"origin + dim * resolution ({expected})"
            )
        if abs(resolution - 0.4) > 1e-9:
            raise ConfigError(f"confirmed_alocc resolution_{axis} must be 0.4 m")


def validate_ros_config(config: dict) -> None:
    for key in ("frame_id", "occupancy_topic", "flow_topic", "marker_topic"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"ros config {key} must be a non-empty string")
    rate = float(config.get("publish_rate_hz", 0))
    if rate <= 0.0 or rate > 20.0:
        raise ConfigError("publish_rate_hz must be in (0, 20]")
    duration = float(config.get("demo_duration_sec", 0))
    if duration <= 0.0:
        raise ConfigError("demo_duration_sec must be positive")
    visualization = config.get("visualization") or {}
    if not isinstance(visualization, dict):
        raise ConfigError("visualization settings must be a mapping")
    stride = int(visualization.get("downsample_stride", 1))
    if stride < 1:
        raise ConfigError("visualization.downsample_stride must be >= 1")
    max_voxels = int(visualization.get("max_voxels", 0))
    if max_voxels < 0:
        raise ConfigError("visualization.max_voxels must be >= 0")
    period = float(visualization.get("min_period_sec", 0.0))
    if period < 0.0:
        raise ConfigError("visualization.min_period_sec must be >= 0")
    selected = visualization.get("classes", [])
    if not isinstance(selected, list) or any(not isinstance(name, str) for name in selected):
        raise ConfigError("visualization.classes must be a list of class names")


def validate_mode(mode: str, model_config: dict) -> None:
    if mode not in ("mock", "file", "python", "ros", "pkl"):
        raise ConfigError("mode must be mock, file, python, ros, or pkl")
    validate_model_config(model_config)
    if mode == "file":
        file_cfg = model_config.get("file") or {}
        path = str(file_cfg.get("path") or "").strip()
        if not path:
            raise ConfigError(
                "File mode needs file.path. Person 2 has not provided prediction files."
            )
        resolved = Path(path)
        if resolved.is_dir():
            pattern = str(file_cfg.get("pattern") or "*.json")
            matches = [item for item in resolved.glob(pattern) if item.is_file()]
            if not matches:
                raise ConfigError(f"No prediction files matching {pattern} in {path}")
        elif not resolved.is_file():
            raise ConfigError(f"Prediction file not found: {path}")
    elif mode == "python":
        python_cfg = model_config.get("python") or {}
        callable_path = str(python_cfg.get("callable") or "").strip()
        if ":" not in callable_path:
            raise ConfigError(
                "Python mode needs python.callable as module:function. "
                "Person 2 has not provided an inference function."
            )
        sample = python_cfg.get("sample", {}) or {}
        if not isinstance(sample, dict):
            raise ConfigError("python.sample must be a mapping")
        sample_path = str(python_cfg.get("sample_path") or "").strip()
        if sample_path:
            resolved = Path(sample_path)
            if resolved.is_dir():
                pattern = str(python_cfg.get("pattern") or "*.json")
                matches = [item for item in resolved.glob(pattern) if item.is_file()]
                if not matches:
                    raise ConfigError(f"No Python samples matching {pattern} in {sample_path}")
            elif not resolved.is_file():
                raise ConfigError(f"python.sample_path not found: {sample_path}")
    elif mode == "ros":
        ros_input = model_config.get("ros_input") or {}
        if not ros_input.get("occupancy_topic") or not ros_input.get("flow_topic"):
            raise ConfigError(
                "ROS input mode needs ros_input.occupancy_topic and ros_input.flow_topic."
            )
        if int(ros_input.get("sync_tolerance_ns", 0)) < 0:
            raise ConfigError("ros_input.sync_tolerance_ns must be >= 0")
        if int(ros_input.get("max_queue", 16)) < 1:
            raise ConfigError("ros_input.max_queue must be >= 1")
        if float(ros_input.get("wait_timeout_sec", 1.0)) < 0.0:
            raise ConfigError("ros_input.wait_timeout_sec must be >= 0")
    elif mode == "pkl":
        pkl_cfg = model_config.get("pkl") or {}
        path = str(pkl_cfg.get("path") or "").strip()
        if not path:
            raise ConfigError("pkl mode needs pkl.path set to a local ALOcc prediction pickle.")
        resolved = Path(path)
        if not resolved.is_file():
            raise ConfigError(f"Prediction pickle not found: {path}")
        if resolved.suffix != ".pkl":
            raise ConfigError(f"pkl mode expects a .pkl file: {path}")
        if int(pkl_cfg.get("max_samples", 0)) < 0:
            raise ConfigError("pkl.max_samples must be >= 0")


def spatial_from_config(model_config: dict, for_flow: bool) -> SpatialMetadata:
    occupancy = model_config["occupancy"]
    flow = model_config.get("flow") or {}
    source = flow if for_flow else occupancy

    def pick(key: str):
        if for_flow and key in flow:
            return flow[key]
        return occupancy[key]

    convention = source.get("coordinate_convention", occupancy["coordinate_convention"])
    return SpatialMetadata(
        dim_x=int(pick("dim_x")),
        dim_y=int(pick("dim_y")),
        dim_z=int(pick("dim_z")),
        resolution_x=float(pick("resolution_x")),
        resolution_y=float(pick("resolution_y")),
        resolution_z=float(pick("resolution_z")),
        origin_x=float(pick("origin_x")),
        origin_y=float(pick("origin_y")),
        origin_z=float(pick("origin_z")),
        coordinate_convention=str(convention),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ALOcc ROS framework configuration.")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--ros-config", required=True)
    args = parser.parse_args(argv)
    try:
        model_config = load_yaml(args.model_config)
        ros_config = load_yaml(args.ros_config)
        validate_ros_config(ros_config)
        validate_mode(args.mode, model_config)
    except (ConfigError, OSError, yaml.YAMLError, TypeError, ValueError) as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1
    print(f"configuration valid for mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
