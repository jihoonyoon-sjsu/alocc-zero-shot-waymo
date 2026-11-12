import pytest

from alocc_ros.config_loader import load_yaml
from alocc_ros.occupancy_geometry import (
    CLASS_COLORS,
    color_rgb,
    marker_options_from_config,
    spatial_from_confirmed,
    voxel_center,
)
from alocc_ros.prediction_types import OccupancyPrediction, index_to_xyz


def _confirmed_spatial(root):
    model = load_yaml(root / "config" / "model.yaml")
    return spatial_from_confirmed(model["confirmed_alocc"]), model


def test_voxel_centers_match_the_confirmed_formula(root):
    spatial, _model = _confirmed_spatial(root)
    assert voxel_center(0, 0, 0, spatial) == pytest.approx((-39.8, -39.8, -0.8))
    assert voxel_center(100, 100, 5, spatial) == pytest.approx((0.2, 0.2, 1.2))
    assert voxel_center(199, 199, 15, spatial) == pytest.approx((39.8, 39.8, 5.2))


def test_linear_index_stays_x_then_y_then_z(root):
    spatial, _model = _confirmed_spatial(root)
    index = 100 + 100 * spatial.dim_x + 5 * spatial.dim_x * spatial.dim_y
    assert index_to_xyz(index, spatial) == (100, 100, 5)
    assert voxel_center(*index_to_xyz(index, spatial), spatial) == pytest.approx((0.2, 0.2, 1.2))
    assert index_to_xyz(0, spatial) == (0, 0, 0)
    last = spatial.voxel_count() - 1
    assert index_to_xyz(last, spatial) == (199, 199, 15)


def test_class_colors_are_stable_and_distinct():
    assert color_rgb("car") == color_rgb("car")
    assert color_rgb("car") == CLASS_COLORS["car"]
    named = ["car", "truck", "bus", "pedestrian", "bicycle", "motorcycle", "barrier", "traffic_cone"]
    colors = [color_rgb(name) for name in named]
    assert len(colors) == len(set(colors))
    assert color_rgb("others") != color_rgb("free")
    assert "free" in CLASS_COLORS


def test_free_space_is_omitted_and_others_is_drawn(root):
    pytest.importorskip("visualization_msgs.msg")
    from alocc_ros.occupancy_geometry import OccupancyMarkerOptions
    from alocc_ros.visualization_node import build_occupancy_markers

    spatial, model = _confirmed_spatial(root)
    # A 2 x 2 x 1 slice of the same ego geometry. Labels use confirmed ids.
    small = spatial.__class__(
        dim_x=2,
        dim_y=2,
        dim_z=1,
        resolution_x=spatial.resolution_x,
        resolution_y=spatial.resolution_y,
        resolution_z=spatial.resolution_z,
        origin_x=spatial.origin_x,
        origin_y=spatial.origin_y,
        origin_z=spatial.origin_z,
        coordinate_convention=spatial.coordinate_convention,
        geometry_confirmed=True,
        max_dimension=256,
    )
    names = tuple(model["confirmed_alocc"]["class_names"])
    # index = x + y * dim_x. (0,0)=others, (1,0)=free, (0,1)=car, (1,1)=free.
    occupancy = OccupancyPrediction(
        spatial=small,
        class_names=names,
        semantic_labels=(0, 17, 4, 17),
        timestamp_sec=0,
        timestamp_nanosec=0,
        is_mock=False,
        source="pkl",
        timestamp_available=False,
        flow_supplied=False,
    )
    options = OccupancyMarkerOptions("free", None, 1, 0)
    built = build_occupancy_markers(occupancy, "ego", options)
    assert built.skipped_free == 2
    assert built.rendered == 2
    assert built.truncated is False
    by_id = {marker.id: marker for marker in built.markers}
    assert set(by_id) == {0, 4}
    assert [(point.x, point.y, point.z) for point in by_id[0].points] == pytest.approx(
        [(-39.8, -39.8, -0.8)]
    )
    assert [(point.x, point.y, point.z) for point in by_id[4].points] == pytest.approx(
        [(-39.8, -39.4, -0.8)]
    )
    assert (by_id[4].color.r, by_id[4].color.g, by_id[4].color.b) == pytest.approx(color_rgb("car"))
    assert (by_id[0].color.r, by_id[0].color.g, by_id[0].color.b) == pytest.approx(color_rgb("others"))


def test_stride_and_cap_are_explicit(root):
    pytest.importorskip("visualization_msgs.msg")
    from alocc_ros.occupancy_geometry import OccupancyMarkerOptions
    from alocc_ros.visualization_node import build_occupancy_markers

    spatial, model = _confirmed_spatial(root)
    small = spatial.__class__(
        dim_x=2,
        dim_y=2,
        dim_z=1,
        resolution_x=spatial.resolution_x,
        resolution_y=spatial.resolution_y,
        resolution_z=spatial.resolution_z,
        origin_x=spatial.origin_x,
        origin_y=spatial.origin_y,
        origin_z=spatial.origin_z,
        coordinate_convention=spatial.coordinate_convention,
        geometry_confirmed=True,
        max_dimension=256,
    )
    names = tuple(model["confirmed_alocc"]["class_names"])
    occupancy = OccupancyPrediction(
        spatial=small,
        class_names=names,
        semantic_labels=(4, 4, 4, 4),
        timestamp_sec=0,
        timestamp_nanosec=0,
        is_mock=False,
        source="pkl",
        timestamp_available=False,
        flow_supplied=False,
    )
    capped = build_occupancy_markers(occupancy, "ego", OccupancyMarkerOptions("free", None, 1, 1))
    assert capped.rendered == 1
    assert capped.truncated is True
    strided = build_occupancy_markers(occupancy, "ego", OccupancyMarkerOptions("free", None, 2, 0))
    assert strided.rendered == 1
    assert strided.skipped_stride == 3
    assert strided.truncated is False
    cars = build_occupancy_markers(
        occupancy,
        "ego",
        OccupancyMarkerOptions("free", frozenset({"pedestrian"}), 1, 0),
    )
    assert cars.rendered == 0
    assert cars.skipped_class == 4


def test_marker_options_read_the_ros_config(root):
    ros_config = load_yaml(root / "config" / "ros2.yaml")
    options = marker_options_from_config(ros_config)
    assert options.free_class == "free"
    assert options.classes is None
    assert options.downsample_stride == 1
    assert options.max_voxels == 0
    assert ros_config["frame_id"] == "ego"
