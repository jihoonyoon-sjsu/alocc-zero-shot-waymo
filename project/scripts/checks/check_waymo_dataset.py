"""CPU check of the Waymo zero-shot config: build the dataset, load two frames
through the test pipeline, and print what the model will receive."""
import sys

import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

cfg = Config.fromfile(sys.argv[1])
if len(sys.argv) > 2:
    cfg.merge_from_dict({'data.test.scene_range': [0, 2]})
cfg.data.test.test_mode = True
ds = build_dataset(cfg.data.test)
print('len', len(ds), 'flags', np.bincount(ds.flag)[:5])
for i in [0, 1, 40]:
    if i >= len(ds):
        continue
    info = ds.get_data_info(i)
    print('--- index', i, 'sample_idx', info['sample_idx'], 'start_of_sequence', info['start_of_sequence'])
    print('curr_to_prev translation', np.round(info['curr_to_prev_ego_rt'][:3, 3].numpy(), 3))
    for name, cam in info['curr']['cams'].items():
        print(f"  {name:16s} {cam['data_path'].split('training/')[1]}  K diag {np.round(np.diag(cam['cam_intrinsic']), 1)} "
              f"c {np.round(cam['cam_intrinsic'][:2, 2], 1)}  cam z-axis in ego {np.round(cam['cam2camego'][:3, 2], 2)}")
    out = ds[i]
    imgs, rots, trans, intrins, post_rots, post_trans, bda = out['img_inputs'][0][0].data if hasattr(out['img_inputs'][0], 'data') else out['img_inputs'][0]
    print('  imgs', tuple(imgs.shape), 'intrins[0]\n', intrins[0].numpy().round(1), '\n  post_rot[0]', post_rots[0].numpy().round(3).tolist(), 'post_tran[0]', post_trans[0].numpy().round(1).tolist())
    print('  img mean per cam', [round(float(imgs[k].mean()), 2) for k in range(imgs.shape[0])])
