"""Fully independent projection: all GT occupied voxels (no mask) onto the raw full-size
images using Waymo v2 camera_calibration (Waymo camera frame: x forward, y left, z up).
Usage: python project_gt_v2_calib.py SCENE FRAME OUT_PNG"""
import glob
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
scene, frame, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
V2 = '/home/jihoon/data/waymo_v2/validation'
cal = np.load(f"/home/jihoon/work/results/figures/v2calib_{scene:03d}.npy", allow_pickle=True).item()

P = '[CameraCalibrationComponent].'
g = np.load(f'/home/jihoon/data/occ3d_waymo/validation/{scene:03d}/{frame:03d}_04.npz')
lab = g['voxel_label']
occ = lab != 23
ii, jj, kk = np.nonzero(occ)
pts = np.stack([(ii + 0.5) * 0.4 - 40, (jj + 0.5) * 0.4 - 40, (kk + 0.5) * 0.4 - 1, np.ones(len(ii))], 1)
L = lab[occ]
col = np.where(L == 1, 'blue', np.where(L == 13, 'gray', np.where(np.isin(L, [3, 5, 6, 10]), 'saddlebrown', np.where(np.isin(L, [11, 12]), 'green', 'magenta'))))
idx = 1000000 + scene * 1000 + frame
fig, ax = plt.subplots(5, 1, figsize=(14, 32))
for r, name in enumerate([1, 2, 3, 4, 5]):
    n = cal['key.camera_name'].index(name)
    ext = np.array(cal[P + 'extrinsic.transform'][n]).reshape(4, 4)
    fu, fv, cu, cv = (cal[P + 'intrinsic.' + k][n] for k in ['f_u', 'f_v', 'c_u', 'c_v'])
    c = pts @ np.linalg.inv(ext).T
    front = c[:, 0] > 0.5
    u = cu - fu * c[front, 1] / c[front, 0]
    v = cv - fv * c[front, 2] / c[front, 0]
    img = Image.open(f'/home/jihoon/data/waymo_alocc/training/image_{name - 1}/{idx}.jpg')
    W, H = img.size
    ins = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    o = np.argsort(-c[front][ins][:, 0])
    ax[r].imshow(img)
    ax[r].scatter(u[ins][o], v[ins][o], s=3, c=col[front][ins][o], alpha=0.5)
    ax[r].set_title(f'camera_name {name} (image_{name - 1}), height {ext[2, 3]:.2f} m: blue vehicle, gray road, brown manmade, green veg, magenta other')
    ax[r].set_xlim(0, W); ax[r].set_ylim(H, 0)
plt.tight_layout(); plt.savefig(out, dpi=40)
print('saved', out)
