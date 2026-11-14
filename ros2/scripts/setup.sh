#!/usr/bin/env bash
# Check the existing ROS 2 Jazzy install and add only lightweight helpers.
set -euo pipefail

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy was not found at /opt/ros/jazzy." >&2
  echo "This script does not install ROS 2." >&2
  exit 1
fi

# shellcheck disable=SC1091
set +u
source /opt/ros/jazzy/setup.bash
set -u
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

echo "ROS_DISTRO=${ROS_DISTRO}"
python3 --version

packages=()
if ! python3 -c "import yaml" >/dev/null 2>&1; then
  packages+=(python3-yaml)
fi
if ! python3 -c "import pytest" >/dev/null 2>&1; then
  packages+=(python3-pytest)
fi
if ! command -v colcon >/dev/null 2>&1; then
  packages+=(python3-colcon-common-extensions)
fi
if ! ros2 pkg prefix visualization_msgs >/dev/null 2>&1; then
  packages+=(ros-jazzy-visualization-msgs)
fi
if ! python3 -c "import numpy" >/dev/null 2>&1; then
  packages+=(python3-numpy)
fi
if ! python3 -c "import PIL" >/dev/null 2>&1; then
  packages+=(python3-pil)
fi

if [[ ${#packages[@]} -gt 0 ]]; then
  echo "Installing lightweight packages: ${packages[*]}"
  sudo apt-get update
  sudo apt-get install -y "${packages[@]}"
else
  echo "Required lightweight packages are already installed."
fi

echo "Setup check complete. ROS 2 Jazzy was left as already installed."
