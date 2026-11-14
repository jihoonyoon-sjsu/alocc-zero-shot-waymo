"""Replaceable prediction input interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from alocc_ros.prediction_types import FlowPrediction, OccupancyPrediction


class ConnectorUnavailable(RuntimeError):
    """The connector exists, but it cannot produce predictions yet."""


class PredictionConnector(ABC):
    """Return occupancy and flow in the common internal format."""

    name: str

    @abstractmethod
    def load(self) -> tuple[OccupancyPrediction, FlowPrediction | None]:
        """Load one occupancy sample and its flow when the source provides flow.

        Flow is None when the source has no usable flow. Callers must not
        replace that with zeros.
        """
