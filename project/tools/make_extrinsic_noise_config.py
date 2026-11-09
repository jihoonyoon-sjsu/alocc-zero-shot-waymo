"""Write a copy of an ALOcc nuScenes config whose camera extrinsics are perturbed.

Noise follows BEVFormer (Li et al., ECCV 2022, Appendix B): at level i, rotation
noise per axis ~ N(0, variance=i) degrees and translation noise per axis
~ N(0, variance=5i) cm. Each camera gets its own noise, drawn once (fixed seed)
and applied to every frame, like a constant calibration error.

The rotation noise is applied in the camera's own frame (R' = R @ dR) and the
translation noise in the ego frame (t' = t + dt).

Usage: python make_extrinsic_noise_config.py BASE_CONFIG OUT_DIR LEVEL [SEED]
Writes OUT_DIR/infos_val_level{LEVEL}.pkl, OUT_DIR/noise_level{LEVEL}.py and
OUT_DIR/noise_level{LEVEL}.json (the sampled noise per camera).
"""
import json
import os
import pickle
import sys

import numpy as np
from mmcv import Config
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

CAMS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def sample_noise(level, seed):
    rng = np.random.default_rng(seed)
    noise = {}
    for cam in CAMS:
        rot_deg = rng.normal(0.0, np.sqrt(level), size=3)
        trans_m = rng.normal(0.0, np.sqrt(5 * level), size=3) / 100.0
        noise[cam] = dict(rot_deg=rot_deg.tolist(), trans_m=trans_m.tolist())
    return noise


def perturb_infos(infos, noise):
    for info in infos:
        for cam, cam_info in info['cams'].items():
            n = noise[cam]
            d_rot = Rotation.from_euler('xyz', n['rot_deg'], degrees=True).as_matrix()
            rot = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix @ d_rot
            cam_info['sensor2ego_rotation'] = list(Quaternion(matrix=rot, atol=1e-6).elements)
            cam_info['sensor2ego_translation'] = list(
                np.asarray(cam_info['sensor2ego_translation']) + np.asarray(n['trans_m']))


def set_ann_file(node, old_name, new_path):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == 'ann_file' and isinstance(v, str) and v.endswith(old_name):
                node[k] = new_path
            else:
                set_ann_file(v, old_name, new_path)
    elif isinstance(node, list):
        for v in node:
            set_ann_file(v, old_name, new_path)


base, out_dir, level = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else level
os.makedirs(out_dir, exist_ok=True)

cfg = Config.fromfile(base)
val_name = os.path.basename(cfg.test_data_config.ann_file)
with open(cfg.test_data_config.ann_file, 'rb') as f:
    data = pickle.load(f)
infos = data['infos'] if isinstance(data, dict) else data

noise = sample_noise(level, seed)
perturb_infos(infos, noise)

pkl_path = os.path.abspath(os.path.join(out_dir, f'infos_val_level{level}.pkl'))
with open(pkl_path, 'wb') as f:
    pickle.dump(data, f)
with open(os.path.join(out_dir, f'noise_level{level}.json'), 'w') as f:
    json.dump(dict(level=level, seed=seed, noise=noise), f, indent=2)

set_ann_file(cfg._cfg_dict, val_name, pkl_path)
cfg.dump(os.path.join(out_dir, f'noise_level{level}.py'))
print('level', level, 'seed', seed, '->', pkl_path)
