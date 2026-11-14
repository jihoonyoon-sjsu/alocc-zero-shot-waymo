# Installation

Target machine: Ubuntu 24.04 with ROS 2 Jazzy already installed. Build this copy from the team repository's `ros2/` directory. Do not build it inside `~/ros2_ws` or the original ALOcc training tree.

## Check and build

```bash
cd /path/to/alocc-zero-shot-waymo/ros2
./scripts/setup.sh
./scripts/build.sh
```

`setup.sh` sources `/opt/ros/jazzy/setup.bash` and stops if Jazzy is missing. It installs a package only when a lightweight dependency is absent:

- `python3-yaml`
- `python3-pytest`
- `python3-numpy`
- `python3-pil` (Pillow, used to decode Waymo JPEG images)
- `python3-colcon-common-extensions`
- `ros-jazzy-visualization-msgs`

NumPy is also used when a saved occupancy pickle is read. Pillow is used by the Waymo scene 0 replay. Both are declared on `alocc_ros` as `exec_depend` entries.

`rviz2` (`ros-jazzy-rviz2`) is optional. Install it only when you want a graphical occupancy view. Headless replay and the test suite do not need it.

It does not install CUDA, the ALOcc training stack, datasets, or checkpoints.

`build.sh` runs that check and then:

```bash
colcon build --symlink-install --packages-select alocc_interfaces alocc_ros
```

## Shell environment

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
```

`run_demo.sh` and the tests set discovery to localhost when it is not already set. That keeps ROS 2 communication inside the VM.

## Resource limits

The checked-in mock grid is 8 x 8 x 4. Mock, JSON, Python, and ROS validation reject a dimension above 64 so a configuration mistake cannot allocate a large voxel volume on a 6 GB virtual machine. Saved ALOcc occupancy is 200 x 200 x 16, so pickle playback allows up to 256 on each axis and still rejects larger axes.
