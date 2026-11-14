"""Build the scene 0 image, camera info, and occupancy messages.

Camera info uses only fields present in ``calibration.json``. This dataset
provides width, height, and K. It does not provide D, R, or P. Those ROS
fields stay at the message defaults: an empty distortion list and zero
rectification and projection matrices. The zeros are not a measured identity
transform.
"""

from __future__ import annotations

from sensor_msgs.msg import CameraInfo, Image

from alocc_ros.message_conversion import occupancy_to_msg
from alocc_ros.waymo_scene0_dataset import (
    EGO_FRAME_ID,
    CameraCalibration,
    ReplayScene,
    SceneFrame,
    occupancy_prediction,
)


def build_image(frame_id: str, sec: int, nanosec: int, height: int, width: int, data: bytes) -> Image:
    message = Image()
    message.header.stamp.sec = sec
    message.header.stamp.nanosec = nanosec
    message.header.frame_id = frame_id
    message.height = height
    message.width = width
    message.encoding = "rgb8"
    message.is_bigendian = 0
    message.step = width * 3
    message.data = data
    return message


def build_camera_info(camera: CameraCalibration, sec: int, nanosec: int) -> CameraInfo:
    message = CameraInfo()
    message.header.stamp.sec = sec
    message.header.stamp.nanosec = nanosec
    message.header.frame_id = camera.name
    message.width = camera.width
    message.height = camera.height
    message.k = list(camera.k)
    if camera.distortion_model is not None:
        message.distortion_model = camera.distortion_model
    if camera.d is not None:
        message.d = list(camera.d)
    if camera.r is not None:
        message.r = list(camera.r)
    if camera.p is not None:
        message.p = list(camera.p)
    return message


def build_occupancy(scene: ReplayScene, frame: SceneFrame):
    return occupancy_to_msg(occupancy_prediction(scene, frame), EGO_FRAME_ID)
