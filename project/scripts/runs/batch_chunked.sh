#!/bin/bash
# Remaining nuScenes runs (baseline, extrinsic noise levels 1-4), each split into
# 3 chunks of ~10 min that save on their own. Safe to relaunch after a restart:
# finished chunks and finished runs are skipped. Pause with pause_batch.sh.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alocc
cd ~/work/ALOcc
TOOLS="/mnt/c/Users/Ji Hoon Yoon/Downloads/school/alocc-zero-shot-waymo/project/tools"
CKPT=ckpts/alocc_3d_r50_256x704_bevdet_preatrain_16f.pth
BASE=configs/alocc/alocc_3d_r50_256x704_bevdet_preatrain_16f.py
N=$HOME/work/results/batch_nuscenes
E=$HOME/work/results/extrinsic_noise
LOG=$HOME/work/results/chunked_progress.txt

job() {  # out_dir run_name full_config
  local D=$1 RUN=$2 CFG=$3 C=$1/chunks
  [ -s "$D/preds/$RUN.pkl" ] && return
  mkdir -p "$C" "$D/preds" "$D/chunk_logs"
  [ -f "$C/${RUN}_chunks.json" ] || python "$TOOLS/make_chunk_configs.py" "$CFG" "$C" "$RUN" 3 >> "$LOG" 2>&1
  for i in 0 1 2; do
    for T in 1 2; do
      [ -s "$D/preds/${RUN}_chunk$i.pkl" ] && break
      echo "=== $RUN chunk$i try$T start $(date)" >> "$LOG"
      bash tools/dist_test.sh "$C/${RUN}_chunk$i.py" "$CKPT" 1 --out "$D/preds/${RUN}_chunk$i.pkl" > "$D/chunk_logs/${RUN}_chunk${i}_try$T.log" 2>&1
      echo "=== $RUN chunk$i try$T exit $? $(date)" >> "$LOG"
    done
  done
  echo "=== $RUN merge start $(date)" >> "$LOG"
  python "$TOOLS/merge_chunks.py" "$C/${RUN}_chunks.json" "$D/preds/${RUN}_chunk{i}.pkl" "$D/preds/$RUN.pkl" > "$D/$RUN.merged.log" 2>&1
  echo "=== $RUN merge exit $? $(date)" >> "$LOG"
}

job "$N" baseline "$BASE"
for L in 1 2 3 4; do
  job "$E" "level$L" "$E/noise_level$L.py"
done
echo "=== ALL DONE $(date)" >> "$LOG"
