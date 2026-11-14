"""Read prediction JSON written to the framework file contract.

`file.path` may be one JSON file or a directory of JSON files. This schema is
the integration contract. It is not an ALOcc, nuScenes, or Waymo file format.
"""

from __future__ import annotations

import json
from pathlib import Path

from alocc_ros.connectors.base import ConnectorUnavailable, PredictionConnector
from alocc_ros.prediction_types import (
    FlowPrediction,
    OccupancyPrediction,
    SpatialMetadata,
    validate_prediction_pair,
)


class FilePredictionConnector(PredictionConnector):
    name = "file"

    def __init__(self, model_config: dict):
        file_cfg = model_config.get("file") or {}
        self._path = str(file_cfg.get("path") or "").strip()
        self._pattern = str(file_cfg.get("pattern") or "*.json")
        self._loop = bool(file_cfg.get("loop", True))
        self._files: list[Path] = []
        self._index = 0
        self._primed = False

    def load(self) -> tuple[OccupancyPrediction, FlowPrediction]:
        if not self._path:
            raise ConnectorUnavailable(
                "File connector is a placeholder. Set file.path to a JSON file "
                "or a directory of JSON files using the documented schema, "
                "or wait for Person 2's format."
            )
        if not self._primed:
            self._files = self._discover()
            self._index = 0
            self._primed = True
        elif self._index >= len(self._files):
            if not self._loop:
                raise ConnectorUnavailable(f"No more prediction files in {self._path}")
            self._files = self._discover()
            self._index = 0
        path = self._files[self._index]
        self._index += 1
        return self._read(path)

    def _discover(self) -> list[Path]:
        path = Path(self._path)
        if path.is_dir():
            matches = sorted(item for item in path.glob(self._pattern) if item.is_file())
            if not matches:
                raise ConnectorUnavailable(
                    f"No prediction files matching {self._pattern} in {path}"
                )
            return matches
        if not path.is_file():
            raise ConnectorUnavailable(f"Prediction file not found: {path}")
        return [path]

    def _read(self, path: Path) -> tuple[OccupancyPrediction, FlowPrediction]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ConnectorUnavailable("Prediction file root must be a JSON object")
            occupancy, flow = predictions_from_placeholder_json(payload)
            validate_prediction_pair(occupancy, flow)
        except ConnectorUnavailable:
            raise
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ConnectorUnavailable(
                f"Prediction file {path} failed validation: {exc}"
            ) from exc
        return occupancy, flow


def predictions_from_placeholder_json(payload: dict) -> tuple[OccupancyPrediction, FlowPrediction]:
    spatial_raw = payload["spatial"]
    spatial = SpatialMetadata(
        dim_x=int(spatial_raw["dim_x"]),
        dim_y=int(spatial_raw["dim_y"]),
        dim_z=int(spatial_raw["dim_z"]),
        resolution_x=float(spatial_raw["resolution_x"]),
        resolution_y=float(spatial_raw["resolution_y"]),
        resolution_z=float(spatial_raw["resolution_z"]),
        origin_x=float(spatial_raw["origin_x"]),
        origin_y=float(spatial_raw["origin_y"]),
        origin_z=float(spatial_raw["origin_z"]),
        coordinate_convention=str(spatial_raw["coordinate_convention"]),
    )
    occupancy = OccupancyPrediction(
        spatial=spatial,
        class_names=tuple(str(name) for name in payload["class_names"]),
        semantic_labels=tuple(int(label) for label in payload["semantic_labels"]),
        timestamp_sec=int(payload["timestamp_sec"]),
        timestamp_nanosec=int(payload["timestamp_nanosec"]),
        is_mock=bool(payload.get("is_mock", False)),
        source="file",
        label_layout=str(payload.get("label_layout", "x_fastest_then_y_then_z")),
    )
    flow = FlowPrediction(
        spatial=spatial,
        flow_vectors=tuple(float(value) for value in payload["flow_vectors"]),
        timestamp_sec=int(payload["timestamp_sec"]),
        timestamp_nanosec=int(payload["timestamp_nanosec"]),
        is_mock=bool(payload.get("is_mock", False)),
        source="file",
        flow_units=str(payload["flow_units"]),
        vector_layout=str(payload.get("vector_layout", "xyz_per_voxel_x_fastest")),
    )
    return occupancy, flow
