"""Check that Occ3D-Waymo's index files (scene index -> frames) line up with our
Waymo v2 parquet download, assuming scene index = position in sorted segment names."""
import glob
import os
import pickle

import numpy as np
import pyarrow.parquet as pq

V2 = os.path.expanduser('~/data/waymo_v2/validation')
OCC = os.path.expanduser('~/data/occ3d_waymo')

segs = sorted(os.path.basename(p)[:-8] for p in glob.glob(f'{V2}/vehicle_pose/*.parquet'))
print('segments in v2 download:', len(segs))
pose_all = pickle.load(open(f'{OCC}/cam_infos_vali.pkl', 'rb'))
print('scenes in cam_infos_vali:', len(pose_all))
print('example columns:', pq.read_schema(f'{V2}/vehicle_pose/{segs[0]}.parquet').names)

occ_dirs = sorted(os.listdir(f'{OCC}/validation'))
print('occ label folders:', len(occ_dirs), occ_dirs[:2])

bad = 0
for scene_idx in [0, 1, 100, 201]:
    t = pq.read_table(f'{V2}/vehicle_pose/{segs[scene_idx]}.parquet').to_pandas()
    t = t.sort_values('key.frame_timestamp_micros')
    col = [c for c in t.columns if 'world_from_vehicle' in c][0]
    poses = np.stack([np.asarray(v, dtype=np.float64).reshape(4, 4) for v in t[col]])
    occ_frames = len(pose_all[scene_idx])
    for f in [0, occ_frames - 1]:
        e = pose_all[scene_idx][f][0]['ego2global']
        # Occ3D frame f may be offset from parquet frame f; find nearest pose.
        d = np.linalg.norm(poses[:, :3, 3] - e[:3, 3], axis=1)
        j = int(d.argmin())
        print(f'scene {scene_idx} {segs[scene_idx][:20]}.. parquet frames {len(poses)} occ frames {occ_frames} '
              f'| occ frame {f} -> parquet frame {j}, dist {d[j]:.3f} m')
        bad += d[j] > 0.05
print('MISMATCHES:', bad)
