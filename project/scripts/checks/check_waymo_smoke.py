"""Sanity check of zero-shot Waymo predictions for scene 0 (frames 0, 5, ..., at 2 Hz):
- geometric IoU (occupied vs free) inside the camera-visible mask, for the
  prediction as-is and flipped/transposed (as-is should be clearly best if the
  grids are aligned),
- what the model predicts on GT road and GT vehicle voxels."""
import sys

import mmcv
import numpy as np

OCC = '/home/jihoon/data/occ3d_waymo/validation'
NUS = ['others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian',
       'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
       'manmade', 'vegetation', 'free']
preds = mmcv.load(sys.argv[1])
by_index = {}
for p in preds:
    by_index.setdefault(p['index'], p['pred_occupancy'])
print('results', len(preds), 'unique frames', len(by_index))
p0 = by_index[0]
print('pred shape', p0.shape, p0.dtype, 'values', np.unique(p0))

variants = {
    'as-is': lambda a: a,
    'flip x': lambda a: a[::-1],
    'flip y': lambda a: a[:, ::-1],
    'transpose xy': lambda a: a.transpose(1, 0, 2),
}
inter = {k: 0 for k in variants}
union = {k: 0 for k in variants}
VEH = [3, 4, 9, 10]  # bus, car, trailer, truck
veh_hit = {k: 0 for k in variants}
veh_all = 0
road_hist = np.zeros(18)
veh_hist = np.zeros(18)
for idx, pred in sorted(by_index.items()):
    frame = idx * 5
    gt = np.load(f'{OCC}/000/{frame:03d}_04.npz')
    # Same mask as ALOcc's Waymo evaluation: in camera view and observed by camera.
    label = gt['voxel_label']
    mask = gt['infov'].astype(bool) & gt['final_voxel_state'].astype(bool)
    gt_occ = (label != 23) & mask
    for k, f in variants.items():
        pr_occ = (f(pred) != 17) & mask
        inter[k] += (pr_occ & gt_occ).sum()
        union[k] += (pr_occ | gt_occ).sum()
        veh_hit[k] += np.isin(f(pred)[(label == 1) & mask], VEH).sum()
    veh_all += ((label == 1) & mask).sum()
    road_hist += np.bincount(pred[(label == 13) & mask], minlength=18)
    veh_hist += np.bincount(pred[(label == 1) & mask], minlength=18)
for k in variants:
    print(f'geometric IoU {k:13s} {100 * inter[k] / union[k]:.1f}   GT vehicle predicted as a vehicle {100 * veh_hit[k] / veh_all:.1f}%')
for name, h in [('GT road', road_hist), ('GT vehicle', veh_hist)]:
    top = np.argsort(h)[::-1][:4]
    print(name, 'predicted as:', ', '.join(f'{NUS[t]} {100 * h[t] / h.sum():.0f}%' for t in top))
