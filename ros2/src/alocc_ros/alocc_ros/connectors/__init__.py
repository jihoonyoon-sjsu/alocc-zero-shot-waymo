"""Prediction connectors. Only the mock connector produces data by default."""

from alocc_ros.connectors.base import ConnectorUnavailable, PredictionConnector
from alocc_ros.connectors.file_input import FilePredictionConnector
from alocc_ros.connectors.mock import MockPredictionConnector
from alocc_ros.connectors.pickle_input import PicklePredictionConnector
from alocc_ros.connectors.python_inference import PythonInferenceConnector
from alocc_ros.connectors.ros_subscription import RosSubscriptionConnector

__all__ = [
    "ConnectorUnavailable",
    "FilePredictionConnector",
    "MockPredictionConnector",
    "PicklePredictionConnector",
    "PredictionConnector",
    "PythonInferenceConnector",
    "RosSubscriptionConnector",
    "create_connector",
]


def create_connector(mode: str, model_config: dict, ros_config: dict, node=None):
    if mode == "mock":
        return MockPredictionConnector(model_config)
    if mode == "file":
        return FilePredictionConnector(model_config)
    if mode == "python":
        return PythonInferenceConnector(model_config)
    if mode == "ros":
        return RosSubscriptionConnector(model_config, ros_config, node)
    if mode == "pkl":
        return PicklePredictionConnector(model_config)
    raise ConnectorUnavailable(f"unknown prediction mode: {mode}")
