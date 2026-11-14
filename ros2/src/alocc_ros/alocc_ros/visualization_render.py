"""Dependency-free SVG slices for mock occupancy and flow."""

from __future__ import annotations

from alocc_ros.occupancy_geometry import color_hex
from alocc_ros.prediction_types import FlowPrediction, OccupancyPrediction, index_to_xyz


def render_occupancy_slice_svg(prediction: OccupancyPrediction, z_index: int = 0, cell: int | None = None) -> str:
    prediction.validate()
    if z_index < 0 or z_index >= prediction.spatial.dim_z:
        raise ValueError(f"z_index {z_index} is outside the occupancy grid")
    if cell is None:
        cell = 3 if max(prediction.spatial.dim_x, prediction.spatial.dim_y) > 32 else 18
    width = prediction.spatial.dim_x * cell + 20
    height = prediction.spatial.dim_y * cell + 36
    kind = "MOCK" if prediction.is_mock else "REAL"
    if prediction.spatial.coordinate_convention == "ego_x_forward_y_left_z_up":
        geometry = " geometry=ego_frame"
    elif prediction.spatial.geometry_confirmed:
        geometry = ""
    else:
        geometry = " geometry=unconfirmed_voxel_indices"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#111111"/>',
        (
            f'<text x="8" y="16" fill="#f2f2f2" font-size="12">'
            f'{kind} occupancy slice z={z_index} source={prediction.source}{geometry}</text>'
        ),
    ]
    for index, label in enumerate(prediction.semantic_labels):
        x, y, z = index_to_xyz(index, prediction.spatial)
        if z != z_index:
            continue
        class_name = prediction.class_names[label] if label < len(prediction.class_names) else ""
        color = color_hex(class_name, label)
        parts.append(
            f'<rect x="{8 + x * cell}" y="{24 + y * cell}" width="{cell - 1}" '
            f'height="{cell - 1}" fill="{color}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_flow_slice_svg(prediction: FlowPrediction, z_index: int = 0, cell: int = 18) -> str:
    prediction.validate()
    if z_index < 0 or z_index >= prediction.spatial.dim_z:
        raise ValueError(f"z_index {z_index} is outside the flow grid")
    width = prediction.spatial.dim_x * cell + 20
    height = prediction.spatial.dim_y * cell + 36
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#111111"/>',
        (
            f'<text x="8" y="16" fill="#f2f2f2" font-size="12">'
            f'MOCK flow slice z={z_index} source={prediction.source}</text>'
        ),
    ]
    stride = prediction.spatial.dim_x * prediction.spatial.dim_y
    for y in range(prediction.spatial.dim_y):
        for x in range(prediction.spatial.dim_x):
            offset = ((z_index * stride) + (y * prediction.spatial.dim_x) + x) * 3
            vx = prediction.flow_vectors[offset]
            vy = prediction.flow_vectors[offset + 1]
            x0 = 8 + x * cell + cell / 2
            y0 = 24 + y * cell + cell / 2
            x1 = x0 + max(-cell + 2, min(cell - 2, vx * 400))
            y1 = y0 + max(-cell + 2, min(cell - 2, -vy * 400))
            parts.append(
                f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
                'stroke="#7fdbff" stroke-width="1.5"/>'
            )
    parts.append("</svg>")
    return "".join(parts)
