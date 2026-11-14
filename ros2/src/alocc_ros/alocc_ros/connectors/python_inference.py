"""Call a Python function supplied by Person 2.

Both of these signatures work::

    def predict(model_config: dict) -> tuple[OccupancyPrediction, FlowPrediction]:
        ...

    def predict(model_config: dict, sample: PredictionSample) -> tuple[OccupancyPrediction, FlowPrediction]:
        ...

A one-argument function still receives only the model config. A function that
accepts a second parameter also receives a `PredictionSample`. This module
does not import or run an ALOcc model.
"""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from alocc_ros.connectors.base import ConnectorUnavailable, PredictionConnector
from alocc_ros.prediction_types import (
    FlowPrediction,
    OccupancyPrediction,
    PredictionValidationError,
    validate_prediction_pair,
)


@dataclass(frozen=True)
class PredictionSample:
    """Sample information handed to a two-argument prediction function.

    `payload` is opaque data for that function. This connector does not read
    it as occupancy or flow.
    """

    sample_id: str = ""
    timestamp_sec: int | None = None
    timestamp_nanosec: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    payload: Mapping[str, Any] = field(default_factory=dict)


class PythonInferenceConnector(PredictionConnector):
    name = "python"

    def __init__(self, model_config: dict, predict_fn=None):
        python_cfg = model_config.get("python") or {}
        self._model_config = model_config
        self._predict_fn = predict_fn
        self._callable_path = str(python_cfg.get("callable") or "").strip()
        self._inline_sample = python_cfg.get("sample", {}) or {}
        self._sample_path = str(python_cfg.get("sample_path") or "").strip()
        self._pattern = str(python_cfg.get("pattern") or "*.json")
        self._loop = bool(python_cfg.get("loop", True))
        self._samples: list[PredictionSample] = []
        self._index = 0
        self._primed = False

    def load(self) -> tuple[OccupancyPrediction, FlowPrediction]:
        predict = self._predict_fn or self._import_callable()
        sample = self._next_sample()
        occupancy, flow = self._invoke(predict, sample)
        if not isinstance(occupancy, OccupancyPrediction) or not isinstance(flow, FlowPrediction):
            raise ConnectorUnavailable(
                "Python inference function must return OccupancyPrediction and FlowPrediction"
            )
        if occupancy.source != "python" or flow.source != "python":
            raise ConnectorUnavailable("Python inference results must set source to 'python'")
        try:
            validate_prediction_pair(occupancy, flow)
        except PredictionValidationError as exc:
            raise ConnectorUnavailable(
                f"Python prediction for sample {sample.sample_id!r} failed validation: {exc}"
            ) from exc
        return occupancy, flow

    def _next_sample(self) -> PredictionSample:
        if not self._primed:
            self._samples = self._discover_samples()
            self._index = 0
            self._primed = True
        elif self._index >= len(self._samples):
            if not self._loop:
                raise ConnectorUnavailable("No more Python samples.")
            self._samples = self._discover_samples()
            self._index = 0
        sample = self._samples[self._index]
        self._index += 1
        return sample

    def _discover_samples(self) -> list[PredictionSample]:
        if not isinstance(self._inline_sample, dict):
            raise ConnectorUnavailable("python.sample must be a mapping")
        base = sample_from_mapping(self._inline_sample)
        if not self._sample_path:
            return [base]
        path = Path(self._sample_path)
        if path.is_file():
            return [merge_samples(base, sample_from_mapping(self._read_json(path)))]
        if not path.is_dir():
            raise ConnectorUnavailable(f"python.sample_path not found: {path}")
        files = sorted(item for item in path.glob(self._pattern) if item.is_file())
        if not files:
            raise ConnectorUnavailable(f"No Python samples matching {self._pattern} in {path}")
        return [merge_samples(base, sample_from_mapping(self._read_json(item))) for item in files]

    def _read_json(self, path: Path) -> dict:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConnectorUnavailable(f"Could not read Python sample {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ConnectorUnavailable(f"Python sample {path} must be a JSON object")
        return payload

    def _invoke(self, predict, sample: PredictionSample):
        try:
            signature = inspect.signature(predict)
        except (TypeError, ValueError):
            try:
                return predict(self._model_config, sample)
            except TypeError:
                return predict(self._model_config)
        try:
            signature.bind(self._model_config, sample)
        except TypeError:
            return predict(self._model_config)
        return predict(self._model_config, sample)

    def _import_callable(self):
        if ":" not in self._callable_path:
            raise ConnectorUnavailable(
                "Python inference connector is a placeholder. Set python.callable "
                "to module:function when Person 2 provides an inference function."
            )
        module_name, function_name = self._callable_path.split(":", 1)
        if not module_name or not function_name:
            raise ConnectorUnavailable("python.callable must look like module:function")
        try:
            module = importlib.import_module(module_name)
            return getattr(module, function_name)
        except (ImportError, AttributeError) as exc:
            raise ConnectorUnavailable(
                f"Could not import inference callable {self._callable_path}: {exc}"
            ) from exc


def sample_from_mapping(data: Mapping[str, Any]) -> PredictionSample:
    if not isinstance(data, Mapping):
        raise ConnectorUnavailable("Python sample must be a mapping")
    known = {"sample_id", "timestamp_sec", "timestamp_nanosec", "metadata", "payload"}
    metadata = data.get("metadata") or {}
    payload = data.get("payload") or {}
    if not isinstance(metadata, Mapping) or not isinstance(payload, Mapping):
        raise ConnectorUnavailable("Python sample metadata and payload must be mappings")
    extra = {key: value for key, value in data.items() if key not in known}
    timestamp_sec = data.get("timestamp_sec")
    timestamp_nanosec = data.get("timestamp_nanosec")
    return PredictionSample(
        sample_id=str(data.get("sample_id") or ""),
        timestamp_sec=None if timestamp_sec is None else int(timestamp_sec),
        timestamp_nanosec=None if timestamp_nanosec is None else int(timestamp_nanosec),
        metadata=dict(metadata),
        payload={**dict(payload), **extra},
    )


def merge_samples(base: PredictionSample, overlay: PredictionSample) -> PredictionSample:
    return PredictionSample(
        sample_id=overlay.sample_id or base.sample_id,
        timestamp_sec=base.timestamp_sec if overlay.timestamp_sec is None else overlay.timestamp_sec,
        timestamp_nanosec=(
            base.timestamp_nanosec if overlay.timestamp_nanosec is None else overlay.timestamp_nanosec
        ),
        metadata={**dict(base.metadata), **dict(overlay.metadata)},
        payload={**dict(base.payload), **dict(overlay.payload)},
    )
