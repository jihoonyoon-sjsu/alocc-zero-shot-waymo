"""Per height bin inside the mask: GT vs predicted share of road / free / occupied (scene 0 frames)."""
import mmcv
import numpy as np
preds = {p['index']: p['pred_occupancy'] for p in mmcv.load('/home/jihoon/work/results/waymo_smoke/scene0.pkl')}
gt_road, pr_road, gt_occ, pr_occ, n = [np.zeros(16) for _ in range(5)]
for i, pred in preds.items():
    g = np.load(f'/home/jihoon/data/occ3d_waymo/validation/000/{i * 5:03d}_04.npz')
    m = g['infov'] & g['final_voxel_state'].astype(bool)
    lab = g['voxel_label']
    n += m.sum((0, 1))
    gt_road += (m & (lab == 13)).sum((0, 1)); pr_road += (m & (pred == 11)).sum((0, 1))
    gt_occ += (m & (lab != 23)).sum((0, 1)); pr_occ += (m & (pred != 17)).sum((0, 1))
print('z bin | masked voxels | GT road | pred road | GT occupied | pred occupied   (shares in %)')
for z in range(16):
    d = max(n[z], 1)
    print(f'{z:2d} ({-1 + 0.4 * z:+.1f}m) {int(n[z]):8d} {100 * gt_road[z] / d:6.1f} {100 * pr_road[z] / d:6.1f} {100 * gt_occ[z] / d:6.1f} {100 * pr_occ[z] / d:6.1f}')
