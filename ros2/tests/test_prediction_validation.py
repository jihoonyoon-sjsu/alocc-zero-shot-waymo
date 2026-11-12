import pytest

from alocc_ros.config_loader import load_yaml, validate_model_config
from alocc_ros.connectors.mock import generate_mock_pair
from alocc_ros.prediction_types import (
    FlowPrediction,
    OccupancyPrediction,
    PredictionValidationError,
    SpatialMetadata,
)


def _spatial(**overrides) -> SpatialMetadata:
    values = dict(
        dim_x=2,
        dim_y=2,
        dim_z=1,
        resolution_x=0.5,
        resolution_y=0.5,
        resolution_z=0.5,
        origin_x=0.0,
        origin_y=0.0,
        origin_z=0.0,
        coordinate_convention="TEST_FRAME",
    )
    values.update(overrides)
    return SpatialMetadata(**values)


def test_valid_mock_predictions_pass(root):
    model = load_yaml(root / "config" / "model.yaml")
    validate_model_config(model)
    occupancy, flow = generate_mock_pair(model, timestamp_sec=1, timestamp_nanosec=2)
    occupancy.validate()
    flow.validate()
    assert occupancy.is_mock is True
    assert flow.is_mock is True
    assert occupancy.source == "mock"
    assert len(occupancy.semantic_labels) == occupancy.spatial.voxel_count()
    assert len(flow.flow_vectors) == flow.spatial.voxel_count() * 3


def test_occupancy_rejects_bad_dimensions_and_labels():
    with pytest.raises(PredictionValidationError):
        _spatial(dim_x=0).validate()
    with pytest.raises(PredictionValidationError):
        _spatial(resolution_z=-1).validate()
    occupancy = OccupancyPrediction(
        spatial=_spatial(),
        class_names=("free", "occupied"),
        semantic_labels=(0, 1, 4, 0),
        timestamp_sec=1,
        timestamp_nanosec=0,
        is_mock=True,
        source="mock",
    )
    with pytest.raises(PredictionValidationError):
        occupancy.validate()


def test_flow_rejects_wrong_vector_length_and_nonfinite_values():
    flow = FlowPrediction(
        spatial=_spatial(),
        flow_vectors=(0.0, 0.0, 0.0),
        timestamp_sec=1,
        timestamp_nanosec=0,
        is_mock=True,
        source="mock",
        flow_units="TEST",
    )
    with pytest.raises(PredictionValidationError):
        flow.validate()
    flow = FlowPrediction(
        spatial=_spatial(),
        flow_vectors=(0.0, 0.0, float("nan")) + (0.0,) * 9,
        timestamp_sec=1,
        timestamp_nanosec=0,
        is_mock=True,
        source="mock",
        flow_units="TEST",
    )
    with pytest.raises(PredictionValidationError):
        flow.validate()
