"""Deterministic synthetic occupancy and flow.

This is mock data for integration testing. It is not ALOcc inference and it
does not read nuScenes or Waymo.
"""

from __future__ import annotations

import time

from alocc_ros.config_loader import spatial_from_config
from alocc_ros.connectors.base import PredictionConnector
from alocc_ros.prediction_types import FlowPrediction, OccupancyPrediction


class MockPredictionConnector(PredictionConnector):
    name = "mock"

    def __init__(self, model_config: dict):
        self._model_config = model_config

    def load(self) -> tuple[OccupancyPrediction, FlowPrediction]:
        return generate_mock_pair(self._model_config)


def generate_mock_pair(
    model_config: dict,
    timestamp_sec: int | None = None,
    timestamp_nanosec: int | None = None,
) -> tuple[OccupancyPrediction, FlowPrediction]:
    """Build a small deterministic pair.

    Voxel values do not depend on the clock. The timestamp defaults to the
    current time so ROS latency can be measured.
    """
    if timestamp_sec is None or timestamp_nanosec is None:
        now = time.time()
        timestamp_sec = int(now)
        timestamp_nanosec = int((now - timestamp_sec) * 1_000_000_000)
    occupancy_spatial = spatial_from_config(model_config, for_flow=False)
    flow_spatial = spatial_from_config(model_config, for_flow=True)
    class_names = tuple(str(name) for name in model_config["occupancy"]["class_names"])
    labels = []
    for z in range(occupancy_spatial.dim_z):
        for y in range(occupancy_spatial.dim_y):
            for x in range(occupancy_spatial.dim_x):
                labels.append((x + 2 * y + 3 * z) % len(class_names))
    vectors = []
    for z in range(flow_spatial.dim_z):
        for y in range(flow_spatial.dim_y):
            for x in range(flow_spatial.dim_x):
                vectors.extend(
                    (
                        0.01 * (x + 1),
                        -0.02 * (y + 1),
                        0.005 * (z + 1),
                    )
                )
    occupancy = OccupancyPrediction(
        spatial=occupancy_spatial,
        class_names=class_names,
        semantic_labels=tuple(labels),
        timestamp_sec=timestamp_sec,
        timestamp_nanosec=timestamp_nanosec,
        is_mock=True,
        source="mock",
        label_layout=str(
            model_config["occupancy"].get("label_layout", "x_fastest_then_y_then_z")
        ),
    )
    flow = FlowPrediction(
        spatial=flow_spatial,
        flow_vectors=tuple(vectors),
        timestamp_sec=timestamp_sec,
        timestamp_nanosec=timestamp_nanosec,
        is_mock=True,
        source="mock",
        flow_units=str(model_config["flow"]["flow_units"]),
        vector_layout=str(model_config["flow"].get("vector_layout", "xyz_per_voxel_x_fastest")),
    )
    occupancy.validate()
    flow.validate()
    return occupancy, flow
