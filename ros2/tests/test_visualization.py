import pytest

from alocc_ros.config_loader import load_yaml
from alocc_ros.connectors.mock import generate_mock_pair
from alocc_ros.visualization_render import render_flow_slice_svg, render_occupancy_slice_svg


def test_svg_slices_are_labelled_mock(root):
    model = load_yaml(root / "config" / "model.yaml")
    occupancy, flow = generate_mock_pair(model, timestamp_sec=1, timestamp_nanosec=0)
    occupancy_svg = render_occupancy_slice_svg(occupancy)
    flow_svg = render_flow_slice_svg(flow)
    assert occupancy_svg.startswith("<svg")
    assert "MOCK occupancy" in occupancy_svg
    assert "MOCK flow" in flow_svg
    assert occupancy_svg.count("<rect") > 10


def test_marker_builders_cover_mock_voxels(root):
    pytest.importorskip("visualization_msgs.msg")
    from visualization_msgs.msg import Marker

    from alocc_ros.occupancy_geometry import OccupancyMarkerOptions
    from alocc_ros.visualization_node import build_flow_markers, build_occupancy_markers

    model = load_yaml(root / "config" / "model.yaml")
    occupancy, flow = generate_mock_pair(model, timestamp_sec=1, timestamp_nanosec=0)
    options = OccupancyMarkerOptions(
        free_class="free", classes=None, downsample_stride=1, max_voxels=0
    )
    built = build_occupancy_markers(occupancy, "ego", options)
    arrows = build_flow_markers(flow, "ego")
    free_index = occupancy.class_names.index("free")
    expected = sum(1 for label in occupancy.semantic_labels if label != free_index)
    rendered = sum(len(marker.points) for marker in built.markers)
    assert built.markers
    assert built.rendered == expected
    assert rendered == expected
    assert built.skipped_free == occupancy.spatial.voxel_count() - expected
    assert all(marker.ns == "mock_occupancy" for marker in built.markers)
    assert all(marker.type == Marker.CUBE_LIST for marker in built.markers)
    assert len(arrows) == flow.spatial.voxel_count()
    assert arrows[0].header.frame_id == "ego"
