#!/bin/bash
# Tara robustness runs on Waymo (single GPU, ~50 min each, ~8 GB GPU).
# Usage: bash ~/work/tara/run_tara.sh [run names...]
#   default runs all six: gaussian_noise_s1..s3 gaussian_blur_s1..s3
# Each run is 4 chunks that save on their own. Safe to stop and relaunch:
# finished chunks and runs are skipped. Scoring (CPU) happens after each run.
RUNS=${@:-"gaussian_noise_s1 gaussian_noise_s2 gaussian_noise_s3 gaussian_blur_s1 gaussian_blur_s2 gaussian_blur_s3"}
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alocc
cd ~/work/ALOcc
P=$HOME/work/project
export PYTHONPATH="$HOME/work/tara:$P/tools:$PYTHONPATH"
CKPT=ckpts/alocc_3d_r50_256x704_bevdet_preatrain_16f.pth
ANN=$HOME/data/waymo_alocc/waymo_infos_val_jpg.pkl
W=$HOME/work/results/tara_robustness
LOG=$W/progress.txt
RANGES="0:51,51:101,101:152,152:202"
mkdir -p "$W/preds" "$W/chunk_logs" "$W/scores"
for RUN in $RUNS; do
  CFG=$HOME/work/tara/configs/$RUN.py
  if [ ! -s "$W/preds/$RUN.pkl" ]; then
    i=0
    for R in ${RANGES//,/ }; do
      for T in 1 2; do
        [ -s "$W/preds/${RUN}_chunk$i.pkl" ] && break
        echo "=== $RUN chunk$i ($R) try$T start $(date)" | tee -a "$LOG"
        bash tools/dist_test.sh "$CFG" "$CKPT" 1 --out "$W/preds/${RUN}_chunk$i.pkl" \
          --cfg-options "data.test.scene_range=[${R/:/,}]" > "$W/chunk_logs/${RUN}_chunk${i}_try$T.log" 2>&1
        echo "=== $RUN chunk$i try$T exit $? $(date)" | tee -a "$LOG"
      done
      i=$((i + 1))
    done
    python "$P/tools/merge_waymo_chunks.py" "$ANN" "$RANGES" "$W/preds/${RUN}_chunk{i}.pkl" "$W/preds/$RUN.pkl" > "$W/$RUN.merged.log" 2>&1
    echo "=== $RUN merged exit $? $(date)" | tee -a "$LOG"
  fi
  [ -s "$W/scores/$RUN.score.json" ] || python "$P/tools/score_waymo.py" "$W/preds/$RUN.pkl" "$W/scores/$RUN.score.json" 2>&1 | grep -v -i warn | tee "$W/scores/$RUN.score.txt"
done
echo "=== ALL DONE $(date)" | tee -a "$LOG"
