"""Check, for all 202 validation scenes:
1. image timestamps (Waymo v2 parquet, sorted segment order) == timestamps in waymo_infos_val (per image_idx);
2. camera extrinsics in Occ3D pose_all slot s == Waymo v2 camera_calibration for the camera
   whose image the pipeline loads into slot s (slot->image file: 0->image_0, 1->image_1, 2->image_3, 3->image_2, 4->image_4;
   image_k holds Waymo camera_name k+1)."""
import glob
import os
import pickle
import numpy as np
import pyarrow.parquet as pq
V2 = '/home/jihoon/data/waymo_v2/validation'
infos = pickle.load(open('/home/jihoon/data/waymo_alocc/waymo_infos_val_jpg.pkl', 'rb'))
ts = {i['image']['image_idx']: i['timestamp'] for i in infos}
pose_all = pickle.load(open('/home/jihoon/data/occ3d_waymo/cam_infos_vali.pkl', 'rb'))
segs = sorted(os.path.basename(p)[:-8] for p in glob.glob(f'{V2}/camera_image/*.parquet'))
slot_to_name = {0: 1, 1: 2, 2: 4, 3: 3, 4: 5}
bad_ts, bad_cal, worst = 0, 0, 0.0
for s, seg in enumerate(segs):
    stamps = sorted(set(pq.read_table(f'{V2}/camera_image/{seg}.parquet', columns=['key.frame_timestamp_micros']).column(0).to_pylist()))
    for f, st in enumerate(stamps):
        if ts.get(1000000 + s * 1000 + f) != st:
            bad_ts += 1
    cal = pq.read_table(f'{V2}/camera_calibration/{seg}.parquet').to_pydict()
    ext = {n: np.array(e).reshape(4, 4) for n, e in zip(cal['key.camera_name'], cal['[CameraCalibrationComponent].extrinsic.transform'])}
    for slot, name in slot_to_name.items():
        d = np.abs(pose_all[s][0][slot]['sensor2ego'] - ext[name]).max()
        worst = max(worst, d)
        if d > 1e-3:
            bad_cal += 1
            if bad_cal <= 5:
                print('calib mismatch scene', s, 'slot', slot, 'cam', name, 'max diff', d)
print(f'scenes {len(segs)}: frames with timestamp mismatch {bad_ts}; slot/calibration mismatches {bad_cal} (worst diff {worst:.2e})')
