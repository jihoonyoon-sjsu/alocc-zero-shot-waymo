import math

import pytest

pytest.importorskip("alocc_interfaces.msg")

from alocc_ros.config_loader import load_yaml
from alocc_ros.connectors.mock import generate_mock_pair
from alocc_ros.message_conversion import (
    flow_from_msg,
    flow_to_msg,
    occupancy_from_msg,
    occupancy_to_msg,
)


def test_message_round_trip_preserves_predictions(root):
    model = load_yaml(root / "config" / "model.yaml")
    occupancy, flow = generate_mock_pair(model, timestamp_sec=11, timestamp_nanosec=12)
    occupancy_msg = occupancy_to_msg(occupancy, "alocc_placeholder")
    flow_msg = flow_to_msg(flow, "alocc_placeholder")
    assert occupancy_msg.header.frame_id == "alocc_placeholder"
    assert occupancy_msg.is_mock is True
    assert list(occupancy_msg.semantic_labels) == list(occupancy.semantic_labels)
    restored_occ = occupancy_from_msg(occupancy_msg)
    restored_flow = flow_from_msg(flow_msg)
    assert restored_occ == occupancy
    assert restored_flow.spatial == flow.spatial
    assert restored_flow.source == flow.source
    assert restored_flow.is_mock == flow.is_mock
    assert restored_flow.flow_units == flow.flow_units
    assert len(restored_flow.flow_vectors) == len(flow.flow_vectors)
    for actual, expected in zip(restored_flow.flow_vectors, flow.flow_vectors):
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
