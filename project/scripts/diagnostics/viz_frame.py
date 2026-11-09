"""Top-down view (highest occupied voxel class) of prediction vs GT, plus a side slice, for one smoke frame."""
import sys
import mmcv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/jihoon/work/project/tools')
from class_mapping import nusc_to_merged, waymo_to_merged, MERGED_CLASSES, OTHER
idx = int(sys.argv[1]); out = sys.argv[2]
pred = {p['index']: p['pred_occupancy'] for p in mmcv.load('/home/jihoon/work/results/waymo_smoke/scene0.pkl')}[idx]
g = np.load(f'/home/jihoon/data/occ3d_waymo/validation/000/{idx * 5:03d}_04.npz')
mask = g['infov'] & g['final_voxel_state'].astype(bool)
P = nusc_to_merged(pred).astype(int); G = waymo_to_merged(g['voxel_label']).astype(int)
P[P == OTHER] = 9; G[G == OTHER] = 9
FREE = MERGED_CLASSES.index('free')
def topdown(a):
    occ = a != FREE
    z = np.where(occ.any(2), 15 - np.argmax(occ[:, :, ::-1], 2), 0)
    img = np.take_along_axis(a, z[:, :, None], 2)[:, :, 0]
    img[~occ.any(2)] = FREE
    return img
cmap = matplotlib.colors.ListedColormap(['blue', 'red', 'orange', 'yellow', 'gray', 'magenta', 'green', 'brown', 'white', 'black'])
fig, ax = plt.subplots(2, 3, figsize=(18, 11))
for a, (t, arr) in zip(ax[0], [('pred (all)', P), ('GT (all)', G), ('pred, GT-masked', np.where(mask, P, FREE))]):
    a.imshow(topdown(arr).T, origin='lower', cmap=cmap, vmin=0, vmax=9); a.set_title(t + '  (x right = array axis 0, y up = axis 1)')
ax[1, 0].imshow(P[:, 100, :].T, origin='lower', cmap=cmap, vmin=0, vmax=9, aspect=4); ax[1, 0].set_title('pred slice y=0 (x vs z)')
ax[1, 1].imshow(G[:, 100, :].T, origin='lower', cmap=cmap, vmin=0, vmax=9, aspect=4); ax[1, 1].set_title('GT slice y=0 (x vs z)')
ax[1, 2].imshow(mask[:, 100, :].T, origin='lower', aspect=4); ax[1, 2].set_title('mask slice y=0')
fig.suptitle('blue vehicle, red ped, orange 2wheel, yellow cone, gray road, magenta sidewalk, green veg, brown manmade, white free, black other')
plt.tight_layout(); plt.savefig(out, dpi=60)
