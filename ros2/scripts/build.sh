#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

"${ROOT}/scripts/setup.sh"

# shellcheck disable=SC1091
set +u
source /opt/ros/jazzy/setup.bash
set -u
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

colcon build --symlink-install --packages-select alocc_interfaces alocc_ros
echo "Build complete. Source ${ROOT}/install/setup.bash before running the demo."
