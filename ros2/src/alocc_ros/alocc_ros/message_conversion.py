"""Convert internal predictions to and from ROS 2 messages."""

from __future__ import annotations

from alocc_interfaces.msg import OccupancyFlow3D, OccupancyGrid3D
from alocc_ros.prediction_types import (
    FlowPrediction,
    OccupancyPrediction,
    SpatialMetadata,
)


def occupancy_to_msg(prediction: OccupancyPrediction, frame_id: str) -> OccupancyGrid3D:
    prediction.validate()
    message = OccupancyGrid3D()
    _fill_header(message, prediction.timestamp_sec, prediction.timestamp_nanosec, frame_id)
    message.is_mock = prediction.is_mock
    message.prediction_source = prediction.source
    message.timestamp_available = prediction.timestamp_available
    message.geometry_confirmed = prediction.spatial.geometry_confirmed
    message.flow_supplied = prediction.flow_supplied
    message.sample_index = prediction.sample_index
    message.sample_metadata = prediction.sample_metadata
    message.pipeline_publish_time_s = 0.0
    _fill_spatial(message, prediction.spatial)
    message.class_names = list(prediction.class_names)
    message.semantic_labels = list(prediction.semantic_labels)
    message.label_layout = prediction.label_layout
    return message


def flow_to_msg(prediction: FlowPrediction, frame_id: str) -> OccupancyFlow3D:
    prediction.validate()
    message = OccupancyFlow3D()
    _fill_header(message, prediction.timestamp_sec, prediction.timestamp_nanosec, frame_id)
    message.is_mock = prediction.is_mock
    message.prediction_source = prediction.source
    message.timestamp_available = True
    message.geometry_confirmed = prediction.spatial.geometry_confirmed
    message.sample_index = 0
    message.sample_metadata = ""
    message.pipeline_publish_time_s = 0.0
    _fill_spatial(message, prediction.spatial)
    message.flow_vectors = [float(value) for value in prediction.flow_vectors]
    message.flow_units = prediction.flow_units
    message.vector_layout = prediction.vector_layout
    return message


def occupancy_from_msg(message: OccupancyGrid3D) -> OccupancyPrediction:
    prediction = OccupancyPrediction(
        spatial=_spatial_from_msg(message),
        class_names=tuple(message.class_names),
        semantic_labels=tuple(int(label) for label in message.semantic_labels),
        timestamp_sec=int(message.header.stamp.sec),
        timestamp_nanosec=int(message.header.stamp.nanosec),
        is_mock=bool(message.is_mock),
        source=str(message.prediction_source),
        label_layout=str(message.label_layout),
        timestamp_available=bool(message.timestamp_available),
        sample_index=int(message.sample_index),
        sample_metadata=str(message.sample_metadata),
        flow_supplied=bool(message.flow_supplied),
    )
    prediction.validate()
    return prediction


def flow_from_msg(message: OccupancyFlow3D) -> FlowPrediction:
    prediction = FlowPrediction(
        spatial=_spatial_from_msg(message),
        flow_vectors=tuple(float(value) for value in message.flow_vectors),
        timestamp_sec=int(message.header.stamp.sec),
        timestamp_nanosec=int(message.header.stamp.nanosec),
        is_mock=bool(message.is_mock),
        source=str(message.prediction_source),
        flow_units=str(message.flow_units),
        vector_layout=str(message.vector_layout),
    )
    prediction.validate()
    return prediction


def _fill_header(message, timestamp_sec: int, timestamp_nanosec: int, frame_id: str) -> None:
    message.header.stamp.sec = int(timestamp_sec)
    message.header.stamp.nanosec = int(timestamp_nanosec)
    message.header.frame_id = frame_id


def _fill_spatial(message, spatial: SpatialMetadata) -> None:
    message.dim_x = spatial.dim_x
    message.dim_y = spatial.dim_y
    message.dim_z = spatial.dim_z
    message.resolution_x = spatial.resolution_x
    message.resolution_y = spatial.resolution_y
    message.resolution_z = spatial.resolution_z
    message.origin_x = spatial.origin_x
    message.origin_y = spatial.origin_y
    message.origin_z = spatial.origin_z
    message.coordinate_convention = spatial.coordinate_convention


def _spatial_from_msg(message) -> SpatialMetadata:
    confirmed = bool(message.geometry_confirmed)
    dims = (int(message.dim_x), int(message.dim_y), int(message.dim_z))
    # Mock, JSON, Python, and ROS publishers still reject an axis above 64
    # before a message exists. A reconstructed ALOcc grid is 200 on x and y,
    # so the reader accepts that saved size without raising the generator cap.
    max_dimension = 256 if max(dims) > 64 else 64
    return SpatialMetadata(
        dim_x=dims[0],
        dim_y=dims[1],
        dim_z=dims[2],
        resolution_x=float(message.resolution_x),
        resolution_y=float(message.resolution_y),
        resolution_z=float(message.resolution_z),
        origin_x=float(message.origin_x),
        origin_y=float(message.origin_y),
        origin_z=float(message.origin_z),
        coordinate_convention=str(message.coordinate_convention),
        geometry_confirmed=confirmed,
        max_dimension=max_dimension,
    )
