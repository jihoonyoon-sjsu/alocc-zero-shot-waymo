#!/bin/bash
# Score nuScenes baseline (8 merged classes) and both Waymo runs, with distance and camera splits. CPU only.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alocc
cd ~/work/ALOcc
P=$HOME/work/project
export PYTHONPATH="$P/tools:$PYTHONPATH"
F="warn|extension|ninja|ldflags|workers"
python "$P/tools/score_nuscenes_merged.py" ~/work/results/batch_nuscenes/preds/baseline.pkl ~/work/results/batch_nuscenes/baseline.merged8.score.json 2>&1 | grep -v -i -E "$F"
W=$HOME/work/results/waymo
for RUN in zero_shot history_reset; do
  python "$P/tools/score_waymo.py" "$W/preds/$RUN.pkl" "$W/$RUN.score.json" 2>&1 | grep -v -i -E "$F"
done
echo "=== SCORING DONE"
