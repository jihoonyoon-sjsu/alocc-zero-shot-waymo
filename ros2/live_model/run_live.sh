#!/bin/bash
# Live ALOcc-3D inside ROS 2 on Waymo scene 0 (40 frames, about 2 minutes, ~8 GB GPU).
# Usage: bash run_live.sh [replay_rate]   (default 0.2 = one frame every 2.5 s)
# Needs: the alocc conda env, ~/work/ALOcc with the checkpoint, ROS 2 Humble,
# and this repo's ros2/ packages built with colcon (set ROS_WS to that workspace).
RATE=${1:-0.2}
HERE=$(cd "$(dirname "$0")" && pwd)
ROS_WS=${ROS_WS:-$HOME/work/ros_live/ws}
CFG=$HERE/../config/model.yaml
DATA=${DATA:-$HOME/alocc-data/waymo_scene0}
cd "$HERE"
source ~/miniforge3/etc/profile.d/conda.sh; conda activate alocc
python alocc_server.py > server.log 2>&1 &
SERVER=$!
until grep -q "model ready" server.log; do sleep 2; kill -0 $SERVER 2>/dev/null || { echo "server failed, see server.log"; exit 1; }; done
conda deactivate
source /opt/ros/humble/setup.bash; source "$ROS_WS/install/setup.bash"
python3 alocc_live_node.py --ros-args -p dataset_path:=$DATA -p model_config:=$CFG > live.log 2>&1 &
LIVE=$!; sleep 4
ros2 run alocc_ros waymo_scene0_replay_node --ros-args -p dataset_path:=$DATA \
  -p replay_rate:=$RATE -p publish_occupancy:=false > replay.log 2>&1 &
wait $LIVE
wait $SERVER
echo "Done. Results in ~/work/ros_live/results (summary.json, per_frame.csv, resource_samples.csv)"
