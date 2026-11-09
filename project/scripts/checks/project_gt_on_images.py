"""Calibration check: project Occ3D-Waymo GT voxels (inside the eval mask) into the
five camera images exactly as the model receives them (resized/cropped 256x704,
same rots/trans/intrins/post_rots/post_trans as the test pipeline).
Usage: python project_gt_on_images.py CONFIG SCENE FRAME_INDEX_IN_SCENE OUT_PNG"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmcv import Config
from mmdet3d.datasets import build_dataset
sys.path.insert(0, '/home/jihoon/work/project/tools')
from class_mapping import waymo_to_merged, OTHER

cfg_file, scene, k, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = Config.fromfile(cfg_file)
cfg.data.test.scene_range = [scene, scene + 1]
ds = build_dataset(cfg.data.test)
data = ds[k]
inputs = data['img_inputs'][0]
inputs = inputs.data if hasattr(inputs, 'data') else inputs
imgs, rots, trans, intrins, post_rots, post_trans = [t for t in inputs[0]][:6] if isinstance(inputs[0], (list, tuple)) else inputs[:6]
frame = k * 5
g = np.load(f'/home/jihoon/data/occ3d_waymo/validation/{scene:03d}/{frame:03d}_04.npz')
mask = g['infov'] & g['final_voxel_state'].astype(bool) & (g['voxel_label'] != 23)
ii, jj, kk = np.nonzero(mask)
pts = np.stack([(ii + 0.5) * 0.4 - 40, (jj + 0.5) * 0.4 - 40, (kk + 0.5) * 0.4 - 1], 1)
cls = waymo_to_merged(g['voxel_label'][mask]).astype(int); cls[cls == OTHER] = 9
colors = np.array(['blue', 'red', 'orange', 'yellow', 'gray', 'magenta', 'green', 'saddlebrown', 'white', 'black'])
mean = np.array([123.675, 116.28, 103.53]); std = np.array([58.395, 57.12, 57.375])
names = cfg.data.test.get('cam_names', None) or ['cam%d' % c for c in range(5)]
fig, ax = plt.subplots(5, 1, figsize=(14, 25))
for c in range(5):
    img = imgs[2 * c].numpy().transpose(1, 2, 0) * std + mean  # imgs alternate current / previous frame per camera
    R, t, K = rots[c].numpy(), trans[c].numpy(), intrins[c].numpy()[:3, :3]
    cam = (pts - t) @ R            # ego -> camera (R is camera->ego rotation)
    front = cam[:, 2] > 0.5
    uv = cam[front] @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    uv = uv @ post_rots[c].numpy()[:2, :2].T + post_trans[c].numpy()[:2]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < 704) & (uv[:, 1] >= 0) & (uv[:, 1] < 256)
    order = np.argsort(-cam[front][inside][:, 2])  # far first, near drawn on top
    ax[c].imshow(np.clip(img / 255, 0, 1))
    ax[c].scatter(uv[inside][order, 0], uv[inside][order, 1], s=3, c=colors[cls[front][inside][order]], alpha=0.6)
    ax[c].set_title(f'scene {scene} frame {frame} slot {c}: GT voxels projected ({inside.sum()} points)')
    ax[c].set_xlim(0, 704); ax[c].set_ylim(256, 0)
fig.suptitle('blue vehicle, red ped, orange 2-wheel, yellow cone, gray road, magenta sidewalk, green veg, brown manmade, black other')
plt.tight_layout(); plt.savefig(out, dpi=55)
print('saved', out)
