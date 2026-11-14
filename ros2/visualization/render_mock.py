#!/usr/bin/env python3
"""Write SVG slices from the mock generator without starting ROS nodes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alocc_ros.config_loader import load_yaml, validate_model_config
from alocc_ros.connectors.mock import generate_mock_pair
from alocc_ros.visualization_render import render_flow_slice_svg, render_occupancy_slice_svg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output-dir", default="results/visualization")
    args = parser.parse_args()
    model_config = load_yaml(args.model_config)
    validate_model_config(model_config)
    occupancy, flow = generate_mock_pair(model_config, timestamp_sec=0, timestamp_nanosec=0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "occupancy_slice.svg").write_text(
        render_occupancy_slice_svg(occupancy), encoding="utf-8"
    )
    (output_dir / "flow_slice.svg").write_text(render_flow_slice_svg(flow), encoding="utf-8")
    print(f"Wrote mock SVG slices to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
