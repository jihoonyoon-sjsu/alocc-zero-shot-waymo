import pytest

from alocc_ros.config_loader import (
    ConfigError,
    load_yaml,
    validate_mode,
    validate_model_config,
    validate_ros_config,
)


def test_repository_configs_load(root):
    model = load_yaml(root / "config" / "model.yaml")
    ros_config = load_yaml(root / "config" / "ros2.yaml")
    validate_model_config(model)
    validate_ros_config(ros_config)
    validate_mode("mock", model)


def test_file_and_python_modes_require_person2_settings(root):
    model = load_yaml(root / "config" / "model.yaml")
    with pytest.raises(ConfigError, match="prediction files"):
        validate_mode("file", model)
    with pytest.raises(ConfigError, match="inference function"):
        validate_mode("python", model)


def test_invalid_rate_is_rejected(root):
    ros_config = load_yaml(root / "config" / "ros2.yaml")
    ros_config["publish_rate_hz"] = 0
    with pytest.raises(ConfigError):
        validate_ros_config(ros_config)


OFFICIAL_CLASS_NAMES = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "free",
]


def test_confirmed_alocc_contract_matches_person2(root):
    model = load_yaml(root / "config" / "model.yaml")
    section = model["confirmed_alocc"]
    names = section["class_names"]
    assert names == OFFICIAL_CLASS_NAMES
    assert len(names) == 18
    assert list(range(18)) == list(range(len(names)))
    assert names[0] == "others"
    assert names[17] == "free"
    assert (section["dim_x"], section["dim_y"], section["dim_z"]) == (200, 200, 16)
    assert section["resolution_x"] == section["resolution_y"] == section["resolution_z"] == 0.4
    assert section["coordinate_frame"] == "ego"
    assert section["axis_order"] == ["x", "y", "z"]
    assert section["axis_directions"] == {"x": "forward", "y": "left", "z": "up"}
    for axis, origin, bound in (
        ("x", -40.0, 40.0),
        ("y", -40.0, 40.0),
        ("z", -1.0, 5.4),
    ):
        assert section[f"origin_{axis}"] == origin
        assert section[f"bound_{axis}_max"] == bound
        assert section[f"origin_{axis}"] + section[f"dim_{axis}"] * section[f"resolution_{axis}"] == pytest.approx(
            bound
        )
    validate_model_config(model)


def test_confirmed_bounds_must_match_voxel_size(root):
    model = load_yaml(root / "config" / "model.yaml")
    model["confirmed_alocc"]["bound_z_max"] = 5.0
    with pytest.raises(ConfigError, match="bound"):
        validate_model_config(model)
