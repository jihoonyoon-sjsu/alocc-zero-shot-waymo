#!/bin/bash
# Zero-shot Waymo runs, after the chunked nuScenes batch finishes (single GPU):
#   zero_shot      normal (16-frame temporal history)
#   history_reset  every frame starts a new sequence (no temporal history)
# Each run is 4 scene-range chunks (~10 min) that save on their own. Safe to
# relaunch after a restart: finished chunks and runs are skipped.
until grep -q "ALL DONE" "$HOME/work/results/chunked_progress.txt" 2>/dev/null; do sleep 60; done
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alocc
cd ~/work/ALOcc
P=$HOME/work/project  # link to the repo's project folder (ALOcc's launcher breaks on spaces)
export PYTHONPATH="$P/tools:$PYTHONPATH"
CFG="$P/configs/alocc_3d_waymo_zero_shot.py"
CKPT=ckpts/alocc_3d_r50_256x704_bevdet_preatrain_16f.pth
ANN=$HOME/data/waymo_alocc/waymo_infos_val_jpg.pkl
W=$HOME/work/results/waymo
LOG=$W/progress.txt
RANGES="0:51,51:101,101:152,152:202"
mkdir -p "$W/preds" "$W/chunk_logs"

job() {  # run_name extra_cfg_options
  local RUN=$1 EXTRA=$2
  [ -s "$W/preds/$RUN.pkl" ] && return
  local i=0
  for R in ${RANGES//,/ }; do
    for T in 1 2; do
      [ -s "$W/preds/${RUN}_chunk$i.pkl" ] && break
      echo "=== $RUN chunk$i ($R) try$T start $(date)" >> "$LOG"
      bash tools/dist_test.sh "$CFG" "$CKPT" 1 --out "$W/preds/${RUN}_chunk$i.pkl" \
        --cfg-options "data.test.scene_range=[${R/:/,}]" $EXTRA > "$W/chunk_logs/${RUN}_chunk${i}_try$T.log" 2>&1
      echo "=== $RUN chunk$i try$T exit $? $(date)" >> "$LOG"
    done
    i=$((i + 1))
  done
  echo "=== $RUN merge start $(date)" >> "$LOG"
  python "$P/tools/merge_waymo_chunks.py" "$ANN" "$RANGES" "$W/preds/${RUN}_chunk{i}.pkl" "$W/preds/$RUN.pkl" > "$W/$RUN.merged.log" 2>&1
  echo "=== $RUN merge exit $? $(date)" >> "$LOG"
}

job zero_shot ""
job history_reset "data.test.sequences_split_num=all"
echo "=== ALL DONE $(date)" >> "$LOG"
