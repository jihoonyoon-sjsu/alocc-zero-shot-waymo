# Testing

Build first so the message package and Python entry points exist:

```bash
cd /path/to/alocc-zero-shot-waymo/ros2
./scripts/build.sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
python3 -m pytest tests -q
```

## Coverage

| Test module | What it checks |
| --- | --- |
| `test_prediction_validation.py` | Valid mock data, rejected dimensions, labels, and flow values |
| `test_mock_generation.py` | Deterministic occupancy and flow formulas, mock labeling |
| `test_config.py` | YAML loading and rejection of unconfigured file or Python modes |
| `test_connectors.py` | Mock output, file placeholder, placeholder JSON, and Python hook |
| `test_prediction_integration.py` | Directory of prediction files, Python sample input, ROS timestamp pairing, and shape checks |
| `test_pickle_input.py` | Synthetic nuScenes and Waymo pickles, missing flow, and occupancy round-trip |
| `test_message_conversion.py` | Internal objects round-trip through both ROS messages |
| `test_ros_pubsub.py` | Publishers and subscribers exchange both outputs; ROS connector converts a pair |
| `test_runtime_monitor.py` | Average, median, 95th percentile, FPS, RSS memory, and CSV context |
| `test_config.py` | YAML loading, the 18-class ego grid, and rejection of a bad bound |
| `test_occupancy_geometry.py` | Voxel centers, `[x, y, z]` order, class colors, and free-space filtering |
| `test_visualization.py` | Mock SVG slices and RViz marker construction |
| `test_waymo_scene0_replay.py` | Waymo scene 0 frame association, timestamps, calibration, and replay timing |

Message and ROS tests skip themselves only when `alocc_interfaces` cannot be imported. After `./scripts/build.sh`, they are expected to run.

## Demo check

```bash
./scripts/run_demo.sh --mode mock
```

The script validates configuration, runs the launch file, and exits with an error if any of these are missing or if validation rows are not all passing:

- `results/runtime/stage_timings.csv`
- `results/runtime/latency.csv`
- `results/runtime/summary.csv`
- `results/runtime/received_occupancy.csv`
- `results/runtime/received_flow.csv`
- `results/runtime/validation.csv`
- `results/visualization/occupancy_slice.svg`
- `results/visualization/flow_slice.svg`

`summary.csv` must contain `mock_pipeline_not_model_inference`.

Real occupancy playback uses a different results directory and does not delete the mock results:

```bash
./scripts/run_demo.sh --mode pkl --input /home/ubuntu/nuscenes_predictions/baseline.pkl --max-samples 8
./scripts/run_demo.sh --mode pkl --input /home/ubuntu/waymo_predictions/zero_shot.pkl --max-samples 8
```

Those runs write `results/real_occupancy/<filename>/`. The summary context is `real_occupancy_pickle_not_model_inference`. Flow is reported as not supplied when the pickle has no usable flow. The occupancy slice is labelled `geometry=ego_frame`. `visualization/marker_summary.txt` records how many free voxels were left out of the RViz cubes.

RViz itself needs a graphical display and `rviz2`. Open `visualization/alocc.rviz` before starting the pickle demo. The fixed frame is `ego` and the marker topic is `/alocc/visualization/markers`. A machine with no display cannot confirm the RViz window. The marker summary and unit tests still check the cube positions and the free-space filter.

Offline SVG rendering, without the launch file:

```bash
source install/setup.bash
python3 visualization/render_mock.py --model-config config/model.yaml
```
