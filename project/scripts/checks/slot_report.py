"""Per pipeline slot: image file, camera name, viewing direction (yaw) and height from the extrinsics the model gets."""
import sys
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
cfg = Config.fromfile('/home/jihoon/work/project/configs/alocc_3d_waymo_zero_shot.py')
cfg.data.test.scene_range = [0, 1]
ds = build_dataset(cfg.data.test)
print('data_config cams:', cfg.data.test.get('data_config', cfg.get('data_config', {})).get('cams'))
info = ds.get_data_info(2)
for name, cam in info['curr']['cams'].items():
    R = cam['cam2camego'][:3, :3]
    z = R[:, 2]  # optical axis in ego frame
    print(f"{name:16s} {cam['data_path'].split('training/')[1]:22s} yaw {np.degrees(np.arctan2(z[1], z[0])):7.1f} deg  height {cam['cam2camego'][2, 3]:.2f} m  K {np.round(cam['cam_intrinsic'][[0, 1, 0, 1], [0, 1, 2, 2]], 0)}")
