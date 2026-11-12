import json

import pytest

from alocc_ros.config_loader import load_yaml
from alocc_ros.connectors.base import ConnectorUnavailable
from alocc_ros.connectors.file_input import FilePredictionConnector
from alocc_ros.connectors.mock import MockPredictionConnector, generate_mock_pair
from alocc_ros.connectors.python_inference import PythonInferenceConnector
from alocc_ros.prediction_types import FlowPrediction, OccupancyPrediction


def test_mock_connector_labels_data_as_mock(root):
    model = load_yaml(root / "config" / "model.yaml")
    occupancy, flow = MockPredictionConnector(model).load()
    assert occupancy.is_mock and flow.is_mock
    assert occupancy.source == "mock"


def test_file_connector_is_unavailable_without_a_path(root):
    model = load_yaml(root / "config" / "model.yaml")
    with pytest.raises(ConnectorUnavailable, match="placeholder"):
        FilePredictionConnector(model).load()


def test_file_connector_reads_placeholder_schema(root, tmp_path):
    model = load_yaml(root / "config" / "model.yaml")
    occupancy, flow = generate_mock_pair(model, timestamp_sec=3, timestamp_nanosec=4)
    payload = {
        "is_mock": False,
        "timestamp_sec": 3,
        "timestamp_nanosec": 4,
        "spatial": occupancy.spatial.__dict__,
        "class_names": list(occupancy.class_names),
        "semantic_labels": list(occupancy.semantic_labels),
        "flow_vectors": list(flow.flow_vectors),
        "flow_units": flow.flow_units,
        "label_layout": occupancy.label_layout,
        "vector_layout": flow.vector_layout,
    }
    path = tmp_path / "prediction.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    model["file"] = {"path": str(path)}
    loaded_occ, loaded_flow = FilePredictionConnector(model).load()
    assert loaded_occ.source == "file"
    assert loaded_occ.is_mock is False
    assert loaded_occ.semantic_labels == occupancy.semantic_labels
    assert loaded_flow.flow_vectors == flow.flow_vectors


def test_python_connector_requires_a_callable(root):
    model = load_yaml(root / "config" / "model.yaml")
    with pytest.raises(ConnectorUnavailable, match="placeholder"):
        PythonInferenceConnector(model).load()


def test_python_connector_accepts_an_injected_function(root):
    model = load_yaml(root / "config" / "model.yaml")

    def predict(model_config):
        occupancy, flow = generate_mock_pair(model_config, timestamp_sec=8, timestamp_nanosec=0)
        return (
            OccupancyPrediction(**{**occupancy.__dict__, "source": "python"}),
            FlowPrediction(**{**flow.__dict__, "source": "python"}),
        )

    occupancy, flow = PythonInferenceConnector(model, predict_fn=predict).load()
    assert occupancy.source == "python"
    assert flow.source == "python"
    occupancy.validate()
    flow.validate()
