#!/bin/bash
# GPU smoke test: zero-shot Waymo on scene 0 only (40 frames), then sanity check.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alocc
cd ~/work/ALOcc
P=$HOME/work/project  # link to the repo's project folder (ALOcc's launcher breaks on spaces)
export PYTHONPATH="$P/tools:$PYTHONPATH"
O=$HOME/work/results/waymo_smoke
mkdir -p "$O"
bash tools/dist_test.sh "$P/configs/alocc_3d_waymo_zero_shot.py" ckpts/alocc_3d_r50_256x704_bevdet_preatrain_16f.pth 1 \
  --out "$O/scene0.pkl" --cfg-options "data.test.scene_range=[0,1]" > "$O/scene0.log" 2>&1
echo "test exit $?"
grep -v -i warn "$O/scene0.log" | grep -E "Error|error|WaymoOccZeroShot|task/s" | tail -5
python /home/jihoon/work/check_waymo_smoke.py "$O/scene0.pkl" 2>&1 | grep -v -i warn
