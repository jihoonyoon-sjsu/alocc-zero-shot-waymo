#!/bin/bash
# Generate extrinsic-noise configs (BEVFormer levels 1-4) and sanity-check level 4.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate alocc
cd ~/work/ALOcc
T="/mnt/c/Users/Ji Hoon Yoon/Downloads/school/alocc-zero-shot-waymo/project/tools"
O=$HOME/work/results/extrinsic_noise
for L in 1 2 3 4; do
  python "$T/make_extrinsic_noise_config.py" configs/alocc/alocc_3d_r50_256x704_bevdet_preatrain_16f.py "$O" $L 2>&1 | grep -v -i warn
done
grep -n "ann_file" "$O/noise_level4.py"
O="$O" python - <<'EOF'
import os, pickle, numpy as np
from pyquaternion import Quaternion
a = pickle.load(open('data/nuscenes/bevdetv2-nuscenes_infos_val.pkl', 'rb'))['infos']
b = pickle.load(open(os.environ['O'] + '/infos_val_level4.pkl', 'rb'))['infos']
print(len(a), len(b))
for i in [0, len(a) - 1]:
    for c in ['CAM_FRONT', 'CAM_BACK']:
        qa = Quaternion(a[i]['cams'][c]['sensor2ego_rotation'])
        qb = Quaternion(b[i]['cams'][c]['sensor2ego_rotation'])
        dt = 100 * (np.array(b[i]['cams'][c]['sensor2ego_translation']) - np.array(a[i]['cams'][c]['sensor2ego_translation']))
        print(i, c, 'angle deg', round(np.degrees((qa.inverse * qb).angle), 3), 'dt cm', np.round(dt, 2))
print('data_path same:', a[0]['cams']['CAM_FRONT']['data_path'] == b[0]['cams']['CAM_FRONT']['data_path'])
EOF
