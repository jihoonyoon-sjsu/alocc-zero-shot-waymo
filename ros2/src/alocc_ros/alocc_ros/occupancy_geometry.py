"""Confirmed ALOcc occupancy geometry and semantic colors.

`config/model.yaml` `confirmed_alocc` is the only numeric source for the
200 x 200 x 16 ego grid. Callers pass that mapping in. This module does not
keep a second copy of the origin, voxel size, or class list.
"""

from __future__ import annotations

from dataclasses import dataclass

from alocc_ros.prediction_types import SpatialMetadata

# Stable colors for the confirmed class names. Unknown mock names use _FALLBACK.
CLASS_COLORS = {
    "others": (0.62, 0.62, 0.62),
    "barrier": (0.95, 0.55, 0.10),
    "bicycle": (0.00, 0.75, 0.85),
    "bus": (0.98, 0.85, 0.12),
    "car": (0.15, 0.35, 0.95),
    "construction_vehicle": (0.72, 0.40, 0.12),
    "motorcycle": (0.78, 0.15, 0.72),
    "pedestrian": (0.90, 0.12, 0.16),
    "traffic_cone": (1.00, 0.28, 0.00),
    "trailer": (0.55, 0.22, 0.72),
    "truck": (0.08, 0.18, 0.55),
    "driveable_surface": (0.55, 0.55, 0.50),
    "other_flat": (0.72, 0.66, 0.48),
    "sidewalk": (0.78, 0.74, 0.68),
    "terrain": (0.42, 0.55, 0.24),
    "manmade": (0.62, 0.48, 0.40),
    "vegetation": (0.18, 0.62, 0.24),
    "free": (0.10, 0.10, 0.10),
}

_FALLBACK = (
    (0.55, 0.55, 0.55),
    (0.24, 0.70, 0.29),
    (0.90, 0.10, 0.29),
    (0.26, 0.39, 0.85),
    (1.00, 0.88, 0.10),
    (0.96, 0.51, 0.19),
)


@dataclass(frozen=True)
class OccupancyMarkerOptions:
    """How many semantic cubes to publish.

    `classes` is None when every class except `free_class` should be drawn.
    `max_voxels` of 0 does not cap the selected voxels.
    """

    free_class: str
    classes: frozenset[str] | None
    downsample_stride: int
    max_voxels: int


def spatial_from_confirmed(section: dict) -> SpatialMetadata:
    """Build ego-frame metadata from the confirmed_alocc mapping."""
    return SpatialMetadata(
        dim_x=int(section["dim_x"]),
        dim_y=int(section["dim_y"]),
        dim_z=int(section["dim_z"]),
        resolution_x=float(section["resolution_x"]),
        resolution_y=float(section["resolution_y"]),
        resolution_z=float(section["resolution_z"]),
        origin_x=float(section["origin_x"]),
        origin_y=float(section["origin_y"]),
        origin_z=float(section["origin_z"]),
        coordinate_convention=str(section["coordinate_convention"]),
        geometry_confirmed=True,
        max_dimension=256,
    )


def voxel_center(i: int, j: int, k: int, spatial: SpatialMetadata) -> tuple[float, float, float]:
    """Return the center of voxel (i, j, k) in the ego frame.

    `i` is x, `j` is y, and `k` is z. The origin is the minimum corner of
    voxel (0, 0, 0), so the center sits half a voxel in from that corner.
    Array axes are not swapped.
    """
    return (
        float(spatial.origin_x) + (i + 0.5) * float(spatial.resolution_x),
        float(spatial.origin_y) + (j + 0.5) * float(spatial.resolution_y),
        float(spatial.origin_z) + (k + 0.5) * float(spatial.resolution_z),
    )


def color_rgb(class_name: str, index: int = 0) -> tuple[float, float, float]:
    """Return a stable RGB color in 0..1 for a semantic class name."""
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    return _FALLBACK[index % len(_FALLBACK)]


def color_hex(class_name: str, index: int = 0) -> str:
    red, green, blue = color_rgb(class_name, index)
    return f"#{int(round(red * 255)):02x}{int(round(green * 255)):02x}{int(round(blue * 255)):02x}"


def marker_options_from_config(ros_config: dict) -> OccupancyMarkerOptions:
    visualization = ros_config.get("visualization") or {}
    selected = visualization.get("classes") or []
    classes = None
    if selected:
        classes = frozenset(str(name) for name in selected)
    return OccupancyMarkerOptions(
        free_class=str(ros_config.get("free_class_name", "free")),
        classes=classes,
        downsample_stride=int(visualization.get("downsample_stride", 1)),
        max_voxels=int(visualization.get("max_voxels", 0)),
    )
