"""Match each camera slot (0-4) in Occ3D-Waymo's cam_infos_vali.pkl to a Waymo
camera name by comparing extrinsic translation and intrinsics with the v2
camera_calibration parquet. Checks several scenes."""
import glob
import os
import pickle

import numpy as np
import pyarrow.parquet as pq

V2 = os.path.expanduser('~/data/waymo_v2/validation')
OCC = os.path.expanduser('~/data/occ3d_waymo')
NAMES = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}

segs = sorted(os.path.basename(p)[:-8] for p in glob.glob(f'{V2}/camera_calibration/*.parquet'))
pose_all = pickle.load(open(f'{OCC}/cam_infos_vali.pkl', 'rb'))
print(pq.read_schema(f'{V2}/camera_calibration/{segs[0]}.parquet').names)

for scene_idx in [0, 57, 201]:
    t = pq.read_table(f'{V2}/camera_calibration/{segs[scene_idx]}.parquet').to_pandas()
    ext_col = [c for c in t.columns if 'extrinsic' in c][0]
    fu_col = [c for c in t.columns if c.endswith('intrinsic.f_u')][0]
    cu_col = [c for c in t.columns if c.endswith('intrinsic.c_u')][0]
    cal = {int(r['key.camera_name']): (np.asarray(r[ext_col]).reshape(4, 4), r[fu_col], r[cu_col])
           for _, r in t.iterrows()}
    out = []
    for slot in range(5):
        p = pose_all[scene_idx][0][slot]
        tr = p['sensor2ego'][:3, 3]
        best = min(cal, key=lambda c: np.linalg.norm(cal[c][0][:3, 3] - tr))
        d = np.linalg.norm(cal[best][0][:3, 3] - tr)
        out.append(f'slot{slot}={NAMES[best]}(d={d:.4f}m, f_u {cal[best][1]:.1f} vs {-p["intrinsics"][0,1]:.1f}, '
                   f'c_u {cal[best][2]:.1f} vs {p["intrinsics"][0,0]:.1f})')
    print('scene', scene_idx, '\n  ' + '\n  '.join(out))
