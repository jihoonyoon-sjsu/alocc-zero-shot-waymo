#!/bin/bash
# Baseline + all 6 single-camera dropouts on Occ3D-nuScenes val, saving predictions.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alocc
cd ~/work/ALOcc
TOOLS="/mnt/c/Users/Ji Hoon Yoon/Downloads/school/alocc-zero-shot-waymo/project/tools"
BASE=configs/alocc/alocc_3d_r50_256x704_bevdet_preatrain_16f.py
CKPT=ckpts/alocc_3d_r50_256x704_bevdet_preatrain_16f.pth
OUT=$HOME/work/results/batch_nuscenes
mkdir -p "$OUT/configs" "$OUT/preds"

run() {  # run_name config_path
  local RUN=$1 CFG=$2
  echo "=== $RUN start $(date)" >> "$OUT/progress.txt"
  bash tools/dist_test.sh "$CFG" "$CKPT" 1 --out "$OUT/preds/$RUN.pkl" > "$OUT/$RUN.log" 2>&1
  echo "=== $RUN exit $? $(date)" >> "$OUT/progress.txt"
}

run baseline "$BASE"
for CAM in CAM_BACK CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK_LEFT CAM_BACK_RIGHT; do
  python "$TOOLS/make_cam_dropout_config.py" "$BASE" "$OUT/configs/drop_$CAM.py" "$CAM" >> "$OUT/progress.txt" 2>&1
  run "drop_$CAM" "$OUT/configs/drop_$CAM.py"
done
echo "=== ALL DONE $(date)" >> "$OUT/progress.txt"
