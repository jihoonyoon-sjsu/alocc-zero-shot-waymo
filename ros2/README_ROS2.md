# ALOcc ROS 2 integration framework

Person 3's ROS 2 Jazzy framework for moving ALOcc occupancy and flow predictions through a replaceable input interface, ROS 2 topics, visualization, and runtime monitoring.

This repository does not train ALOcc, preprocess nuScenes or Waymo, or download model checkpoints. The working demonstration uses a small deterministic **mock** generator. Mock measurements are not model inference performance.

## Layout

```text
ros2/
├── src/alocc_interfaces/    # OccupancyGrid3D and OccupancyFlow3D
├── src/alocc_ros/           # rclpy nodes, connectors, validation
├── config/                  # mock grid, confirmed ALOcc occupancy, ROS settings
├── scripts/                 # setup, build, and mock demo
├── visualization/           # SVG renderer and optional RViz config
├── results/                 # generated CSV and SVG output
├── tests/
└── docs/
```

## Run the mock demo

From this directory, on the Ubuntu 24.04 VM where ROS 2 Jazzy is already installed:

```bash
./scripts/build.sh
./scripts/run_demo.sh --mode mock
```

The demo does not need a GPU, nuScenes, Waymo, or a checkpoint. It publishes `/alocc/occupancy` and `/alocc/flow`, checks both messages, writes SVG slices, and saves runtime CSV files under `results/runtime/`.

Optional RViz view, when a display is available. Start RViz before the demo so it can receive the volatile marker topic. Fixed frame is `ego`.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
rviz2 -d visualization/alocc.rviz
```

Real occupancy, in a second terminal:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
./scripts/run_demo.sh --mode pkl --input /home/ubuntu/nuscenes_predictions/baseline.pkl --max-samples 8
```

The headless demo still writes `results/visualization/occupancy_slice.svg`. The mock demo also writes `flow_slice.svg`. Real pickle playback does not write a flow slice.

## Replay Waymo scene 0

This is a separate playback of the extracted scene. It does not change `/alocc/occupancy` or `/alocc/visualization/markers`.

The dataset is not part of this repository. Extract it to `~/alocc-data/waymo_scene0`. On the Ubuntu VM that path is `/home/ubuntu/alocc-data/waymo_scene0`. A teammate with another home directory can keep the same folder name under their own home, or pass a different root:

```bash
ros2 launch alocc_ros waymo_scene0_replay.launch.py dataset_root:=/path/to/waymo_scene0
```

The extracted folder needs these files and camera directories:

```text
frames.json
calibration.json
check/zero_shot_scene0.npz
images/CAM_SIDE_LEFT/      # 40 JPEG images
images/CAM_FRONT_LEFT/
images/CAM_FRONT/
images/CAM_FRONT_RIGHT/
images/CAM_SIDE_RIGHT/
```

`check/zero_shot_scene0.npz` supplies the 40 occupancy grids. The large `zero_shot.pkl` is not required for this replay. JPEG decoding uses Pillow, and the occupancy arrays use NumPy. `./scripts/setup.sh` installs `python3-pil` and `python3-numpy` when they are missing.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 launch alocc_ros waymo_scene0_replay.launch.py
```

Camera images, camera info, and the ego-frame occupancy grid share each frame's `timestamp_us` from `frames.json`. The replay launch does not start RViz. A headless SSH session can run this command. `DISPLAY` is not required.

The visualization launch repeats the replay, draws markers on `/alocc/waymo_scene0/markers`, and opens `visualization/waymo_scene0.rviz` when `DISPLAY` is set. If `DISPLAY` is unset, it logs that RViz was skipped and the replay still runs. `rviz2` is optional and is not required for tests or headless playback. The default marker file, `config/waymo_scene0_rviz.yaml`, keeps every second voxel index so the view is lighter than the full cube list. `full_resolution:=true` uses every non-free voxel. Closing RViz stops the launch.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 launch alocc_ros waymo_scene0_visualization.launch.py
```

## Run the tests

From the team repository's `ros2/` directory, after the build and with the Jazzy and workspace setup files sourced:

```bash
python3 -m pytest tests -q
```

The Waymo scene 0 tests read `~/alocc-data/waymo_scene0` when that directory is present. They skip the dataset checks when it is absent. Set `WAYMO_SCENE0_DIR` to point them at another extract.

## What is configurable

`config/model.yaml` `occupancy` is the small mock grid. `confirmed_alocc` is the shared ego-frame contract for saved ALOcc occupancy: 18 nuScenes class names, a 200 x 200 x 16 grid, and 0.4 m voxels. The motion-flow head is outside the current project scope. Mock flow remains a separate demonstration.

## Integration options

Person 2 can later connect real predictions in one of three ways. Only the mock path runs out of the box.

| Mode | Command | Status |
| --- | --- | --- |
| Mock generator | `./scripts/run_demo.sh --mode mock` | Implemented |
| Prediction files | `./scripts/run_demo.sh --mode file --model-config ...` | One JSON file or a directory of JSON files |
| Python function | `./scripts/run_demo.sh --mode python --model-config ...` | `predict(config)` or `predict(config, sample)` |
| ROS 2 input topics | `./scripts/run_demo.sh --mode ros --model-config ...` | Timestamp-paired `OccupancyGrid3D` and `OccupancyFlow3D` |
| Saved ALOcc pickle | `./scripts/run_demo.sh --mode pkl --input PICKLE` | Occupancy from a nuScenes or Waymo prediction pickle. Missing flow is not invented |

Details are in `docs/prediction_interface.md` and `docs/model_integration.md`.

## Existing environment

Build this project from the team repository's `ros2/` directory:

```bash
cd /path/to/alocc-zero-shot-waymo/ros2
./scripts/build.sh
```

Do not rebuild it inside `~/ros2_ws` or the original ALOcc training tree. `scripts/setup.sh` checks for Jazzy and installs only missing lightweight packages such as PyYAML, pytest, NumPy, or Pillow. It does not reinstall ROS 2. The Waymo dataset remains `~/alocc-data/waymo_scene0` and is not stored in this repository.
