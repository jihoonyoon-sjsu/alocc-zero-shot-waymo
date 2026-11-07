#!/bin/bash
# Usage: bash ~/work/run_alocc_test.sh [3d|2d_mini]
set -e
MODEL=${1:-3d}
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alocc
cd ~/work/ALOcc
NAME=alocc_${MODEL}_r50_256x704_bevdet_preatrain_16f
mkdir -p ~/work/results
bash tools/dist_test.sh configs/alocc/$NAME.py ckpts/$NAME.pth 1 2>&1 | tee ~/work/results/${NAME}_$(date +%Y%m%d_%H%M).log
