"""Where does the model predict occupied where GT says free? By distance and height, scenes 0 and 37."""
import pickle
import mmcv
import numpy as np
W = '/home/jihoon/work/results/waymo/preds/zero_shot_chunk0.pkl'
ann = pickle.load(open('/home/jihoon/data/waymo_alocc/waymo_infos_val_jpg.pkl', 'rb'))
import sys; sys.path.insert(0, '/home/jihoon/work/project/tools')
from waymo_zero_shot import keep_frame, scene_frame
frames = [scene_frame(i) for i in sorted(ann, key=lambda i: i['image']['image_idx']) if keep_frame(i, 5, (0, 51))]
preds = {p['index']: p['pred_occupancy'] for p in mmcv.load(W)}
xs = (np.arange(200) + 0.5) * 0.4 - 40
dist = np.sqrt(xs[:, None] ** 2 + xs[None, :] ** 2)[:, :, None].repeat(16, 2)
z = np.broadcast_to(((np.arange(16) + 0.5) * 0.4 - 1)[None, None], (200, 200, 16))
for scene in [0, 37]:
    fp_d, tot_d, fp_z, tot_z = np.zeros(4), np.zeros(4), np.zeros(16), np.zeros(16)
    for i, (s, f) in enumerate(frames):
        if s != scene:
            continue
        g = np.load(f'/home/jihoon/data/occ3d_waymo/validation/{s:03d}/{f:03d}_04.npz')
        m = g['infov'] & g['final_voxel_state'].astype(bool) & (g['voxel_label'] == 23)
        fp = m & (preds[i] != 17)
        b = np.minimum((dist // 10).astype(int), 3)
        fp_d += np.bincount(b[fp], minlength=4); tot_d += np.bincount(b[m], minlength=4)
        zi = np.broadcast_to(np.arange(16), (200, 200, 16))
        fp_z += np.bincount(zi[fp], minlength=16); tot_z += np.bincount(zi[m], minlength=16)
    print(f'scene {scene}: GT-free voxels predicted occupied, by distance 0-10/10-20/20-30/30+ m:',
          ' '.join(f'{100 * a / max(t, 1):.0f}%' for a, t in zip(fp_d, tot_d)))
    print('   by height bin (z=-1..5.4m, 0.4m):', ' '.join(f'{100 * a / max(t, 1):.0f}' for a, t in zip(fp_z, tot_z)))
