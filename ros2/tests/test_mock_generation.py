from alocc_ros.config_loader import load_yaml
from alocc_ros.connectors.mock import generate_mock_pair


def test_mock_occupancy_and_flow_are_deterministic(root):
    model = load_yaml(root / "config" / "model.yaml")
    first_occ, first_flow = generate_mock_pair(model, timestamp_sec=5, timestamp_nanosec=6)
    second_occ, second_flow = generate_mock_pair(model, timestamp_sec=5, timestamp_nanosec=6)
    assert first_occ.semantic_labels == second_occ.semantic_labels
    assert first_flow.flow_vectors == second_flow.flow_vectors
    assert first_occ.semantic_labels[0] == 0
    assert first_flow.flow_vectors[0:3] == (0.01, -0.02, 0.005)
    assert "MOCK" not in first_occ.source
    assert first_occ.source == "mock"
    assert first_occ.is_mock and first_flow.is_mock
