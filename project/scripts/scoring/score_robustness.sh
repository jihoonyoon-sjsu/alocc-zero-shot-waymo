#!/bin/bash
# Score the camera-drop and extrinsic-noise runs on the 8 merged classes with distance and camera-sector splits. CPU only.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alocc
cd ~/work/ALOcc
P=$HOME/work/project
export PYTHONPATH="$P/tools:$PYTHONPATH"
F="warn|extension|ninja|ldflags|workers"
OUT=$HOME/work/results/robustness_merged8
mkdir -p "$OUT"
for PKL in ~/work/results/batch_nuscenes/preds/drop_*.pkl $(ls ~/work/results/extrinsic_noise/preds/level?.pkl ~/work/results/extrinsic_noise/preds/*merged*.pkl 2>/dev/null); do
  RUN=$(basename "$PKL" .pkl)
  [ -f "$OUT/$RUN.score.json" ] && { echo "skip $RUN"; continue; }
  echo "=== $RUN"
  python "$P/tools/score_nuscenes_merged.py" "$PKL" "$OUT/$RUN.score.json" 2>&1 | grep -v -i -E "$F" | tee "$OUT/$RUN.score.txt"
done
echo "=== ROBUSTNESS SCORING DONE"
