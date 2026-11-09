"""Same height profile as z_profile.py, on nuScenes baseline predictions (first 400 frames of chunk 0)."""
import glob
import pickle
import mmcv
import numpy as np
infos = sorted(pickle.load(open('/home/jihoon/work/ALOcc/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl', 'rb'))['infos'], key=lambda e: e['timestamp'])
f = sorted(glob.glob('/home/jihoon/work/results/batch_nuscenes/preds/baseline_chunk0.pkl'))[0]
preds = {}
for p in mmcv.load(f):
    if p['index'] < 400:
        preds.setdefault(p['index'], p['pred_occupancy'])
gt_road, pr_road, gt_occ, pr_occ, n = [np.zeros(16) for _ in range(5)]
for i, pred in preds.items():
    info = infos[i]
    g = np.load(f'/home/jihoon/work/ALOcc/data/nuscenes/gts/{info["scene_name"]}/{info["token"]}/labels.npz')
    m = g['mask_camera'].astype(bool); lab = g['semantics']
    n += m.sum((0, 1))
    gt_road += (m & (lab == 11)).sum((0, 1)); pr_road += (m & (pred == 11)).sum((0, 1))
    gt_occ += (m & (lab != 17)).sum((0, 1)); pr_occ += (m & (pred != 17)).sum((0, 1))
print('nuScenes, frames', len(preds))
print('z bin | masked voxels | GT road | pred road | GT occupied | pred occupied   (shares in %)')
for z in range(16):
    d = max(n[z], 1)
    print(f'{z:2d} ({-1 + 0.4 * z:+.1f}m) {int(n[z]):8d} {100 * gt_road[z] / d:6.1f} {100 * pr_road[z] / d:6.1f} {100 * gt_occ[z] / d:6.1f} {100 * pr_occ[z] / d:6.1f}')
