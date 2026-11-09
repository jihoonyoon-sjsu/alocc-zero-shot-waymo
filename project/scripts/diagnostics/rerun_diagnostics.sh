#!/bin/bash
# Rerun every CPU diagnostic and save its output, so each number in the report has a saved source.
source ~/miniforge3/etc/profile.d/conda.sh
cd ~/work/ALOcc
export PYTHONPATH=/home/jihoon/work/project/tools
O=~/work/results/diagnostics
mkdir -p $O
F="warn|extension|ninja|ldflags|workers"
run() {  # env name output_name command...
  local ENV=$1 NAME=$2; shift 2
  conda activate $ENV
  { echo "# $(date)  env=$ENV  cmd: $*"; "$@" 2>&1 | grep -v -i -E "$F"; } > "$O/$NAME.txt"
  echo "$NAME done"
}
W=/home/jihoon/work
run alocc smoke_check_scene0 python $W/check_waymo_smoke.py ~/work/results/waymo_smoke/scene0.pkl
run alocc height_profile_waymo_scene0 python $W/z_profile.py
run alocc height_profile_nuscenes_400frames python $W/z_profile_nus.py
run alocc scene_spread_and_class_shares python $W/scene_spread.py
run alocc gt_mask_values python $W/inspect_masks.py
run alocc false_occupied_by_distance_height python $W/where_false_occ.py
run alocc history_reset_vs_zero_shot_chunk0 python $W/check_reset_differs.py
run alocc camera_slots python $W/slot_report.py
run alocc all_run_miou bash $W/list_miou.sh
run waymo image_and_calibration_pairing_all_scenes python $W/check_pairing.py
echo "=== DIAGNOSTICS DONE"
