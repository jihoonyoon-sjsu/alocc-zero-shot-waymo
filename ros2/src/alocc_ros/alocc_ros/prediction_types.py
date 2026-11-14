"""Internal occupancy and flow predictions.

Dimensions, class names, units, and coordinate conventions are configuration.
They are not claimed to be the modified ALOcc output contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class PredictionValidationError(ValueError):
    """Raised when an occupancy or flow prediction fails structural checks."""


ALLOWED_SOURCES = ("mock", "file", "python", "ros", "pkl")
MEASUREMENT_CONTEXT = "mock_pipeline_not_model_inference"
REAL_OCCUPANCY_CONTEXT = "real_occupancy_pickle_not_model_inference"


def measurement_context_for_mode(mode: str) -> str:
    """Name the experiment that produced a timing row.

    Mock, JSON, Python, and ROS timings keep the original mock-pipeline label.
    Pickle playback is saved occupancy from an earlier ALOcc run, not inference.
    """
    if mode == "pkl":
        return REAL_OCCUPANCY_CONTEXT
    return MEASUREMENT_CONTEXT


@dataclass(frozen=True)
class SpatialMetadata:
    dim_x: int
    dim_y: int
    dim_z: int
    resolution_x: float
    resolution_y: float
    resolution_z: float
    origin_x: float
    origin_y: float
    origin_z: float
    coordinate_convention: str
    geometry_confirmed: bool = True
    max_dimension: int = 64

    def voxel_count(self) -> int:
        return self.dim_x * self.dim_y * self.dim_z

    def validate(self) -> None:
        if isinstance(self.max_dimension, bool) or not isinstance(self.max_dimension, int):
            raise PredictionValidationError("max_dimension must be a positive integer")
        if self.max_dimension <= 0:
            raise PredictionValidationError("max_dimension must be a positive integer")
        for name in ("dim_x", "dim_y", "dim_z"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise PredictionValidationError(f"{name} must be a positive integer")
            if value > self.max_dimension:
                raise PredictionValidationError(
                    f"{name}={value} exceeds the lightweight limit of {self.max_dimension}"
                )
        if self.geometry_confirmed:
            for name in ("resolution_x", "resolution_y", "resolution_z"):
                value = float(getattr(self, name))
                if not math.isfinite(value) or value <= 0.0:
                    raise PredictionValidationError(f"{name} must be a positive finite number")
            for name in ("origin_x", "origin_y", "origin_z"):
                value = float(getattr(self, name))
                if not math.isfinite(value):
                    raise PredictionValidationError(f"{name} must be finite")
        if not isinstance(self.coordinate_convention, str) or not self.coordinate_convention.strip():
            raise PredictionValidationError("coordinate_convention must be a non-empty string")


@dataclass(frozen=True)
class OccupancyPrediction:
    spatial: SpatialMetadata
    class_names: tuple[str, ...]
    semantic_labels: tuple[int, ...]
    timestamp_sec: int
    timestamp_nanosec: int
    is_mock: bool
    source: str
    label_layout: str = "x_fastest_then_y_then_z"
    timestamp_available: bool = True
    sample_index: int = 0
    sample_metadata: str = ""
    flow_supplied: bool = True

    def validate(self) -> None:
        self.spatial.validate()
        _validate_time_and_source(self.timestamp_sec, self.timestamp_nanosec, self.source)
        if not self.class_names:
            raise PredictionValidationError("class_names must not be empty")
        if any(not isinstance(name, str) or not name.strip() for name in self.class_names):
            raise PredictionValidationError("every class name must be a non-empty string")
        if len(set(self.class_names)) != len(self.class_names):
            raise PredictionValidationError("class_names must be unique")
        if len(self.class_names) > 256:
            raise PredictionValidationError("class_names cannot exceed 256 entries for uint8 labels")
        expected = self.spatial.voxel_count()
        label_shape = (self.spatial.dim_x, self.spatial.dim_y, self.spatial.dim_z)
        if len(self.semantic_labels) != expected:
            raise PredictionValidationError(
                "semantic_labels length "
                f"{len(self.semantic_labels)} != shape {label_shape} ({expected} voxels)"
            )
        class_count = len(self.class_names)
        for index, label in enumerate(self.semantic_labels):
            if isinstance(label, bool) or not isinstance(label, int):
                raise PredictionValidationError(f"semantic label at {index} must be an integer")
            if label < 0 or label >= class_count:
                raise PredictionValidationError(
                    f"semantic label {label} at {index} is outside class_names"
                )
        if self.label_layout != "x_fastest_then_y_then_z":
            raise PredictionValidationError(
                "only label_layout x_fastest_then_y_then_z is implemented"
            )


@dataclass(frozen=True)
class FlowPrediction:
    spatial: SpatialMetadata
    flow_vectors: tuple[float, ...]
    timestamp_sec: int
    timestamp_nanosec: int
    is_mock: bool
    source: str
    flow_units: str
    vector_layout: str = "xyz_per_voxel_x_fastest"

    def validate(self) -> None:
        self.spatial.validate()
        _validate_time_and_source(self.timestamp_sec, self.timestamp_nanosec, self.source)
        expected = self.spatial.voxel_count() * 3
        flow_shape = (self.spatial.dim_x, self.spatial.dim_y, self.spatial.dim_z, 3)
        if len(self.flow_vectors) != expected:
            raise PredictionValidationError(
                f"flow_vectors length {len(self.flow_vectors)} != shape {flow_shape} ({expected} values)"
            )
        for index, value in enumerate(self.flow_vectors):
            number = float(value)
            if not math.isfinite(number):
                raise PredictionValidationError(f"flow value at {index} is not finite")
        if not isinstance(self.flow_units, str) or not self.flow_units.strip():
            raise PredictionValidationError("flow_units must be a non-empty string")
        if self.vector_layout != "xyz_per_voxel_x_fastest":
            raise PredictionValidationError(
                "only vector_layout xyz_per_voxel_x_fastest is implemented"
            )


def _validate_time_and_source(timestamp_sec: int, timestamp_nanosec: int, source: str) -> None:
    if isinstance(timestamp_sec, bool) or not isinstance(timestamp_sec, int) or timestamp_sec < 0:
        raise PredictionValidationError("timestamp_sec must be a non-negative integer")
    if (
        isinstance(timestamp_nanosec, bool)
        or not isinstance(timestamp_nanosec, int)
        or timestamp_nanosec < 0
        or timestamp_nanosec >= 1_000_000_000
    ):
        raise PredictionValidationError("timestamp_nanosec must be in [0, 1000000000)")
    if source not in ALLOWED_SOURCES:
        raise PredictionValidationError(
            f"source must be one of {', '.join(ALLOWED_SOURCES)}"
        )


def stamp_to_ns(timestamp_sec: int, timestamp_nanosec: int) -> int:
    return timestamp_sec * 1_000_000_000 + timestamp_nanosec


def validate_prediction_pair(
    occupancy: OccupancyPrediction,
    flow: FlowPrediction,
    timestamp_tolerance_ns: int = 0,
) -> None:
    """Validate both predictions and require their timestamps to agree.

    Occupancy and flow may use different grids. Each one is checked against
    its own dimensions, label list, and vector layout.
    """
    occupancy.validate()
    flow.validate()
    if isinstance(timestamp_tolerance_ns, bool) or not isinstance(timestamp_tolerance_ns, int):
        raise PredictionValidationError("timestamp tolerance must be an integer number of nanoseconds")
    if timestamp_tolerance_ns < 0:
        raise PredictionValidationError("timestamp tolerance must be >= 0")
    delta = abs(
        stamp_to_ns(occupancy.timestamp_sec, occupancy.timestamp_nanosec)
        - stamp_to_ns(flow.timestamp_sec, flow.timestamp_nanosec)
    )
    if delta > timestamp_tolerance_ns:
        raise PredictionValidationError(
            "occupancy and flow timestamps differ by "
            f"{delta} ns; tolerance is {timestamp_tolerance_ns} ns"
        )


def index_to_xyz(index: int, spatial: SpatialMetadata) -> tuple[int, int, int]:
    """Decode an x-fastest linear index."""
    if index < 0 or index >= spatial.voxel_count():
        raise PredictionValidationError(f"voxel index {index} is out of range")
    x = index % spatial.dim_x
    y = (index // spatial.dim_x) % spatial.dim_y
    z = index // (spatial.dim_x * spatial.dim_y)
    return x, y, z
