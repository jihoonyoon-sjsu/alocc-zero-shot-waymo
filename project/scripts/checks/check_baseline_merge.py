"""One-off check that chunking does not change results: rebuild the full run's
padding (frames 5979-5995 repeated, as in every full ALOcc run here) on the merged
baseline and score it. It should give exactly the earlier full-run 45.46 mIoU."""
import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset

merged = mmcv.load('/home/jihoon/work/results/batch_nuscenes/preds/baseline.pkl')
padded = merged + [dict(merged[i]) for i in range(5979, 5996)]
cfg = Config.fromfile('configs/alocc/alocc_3d_r50_256x704_bevdet_preatrain_16f.py')
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)
eval_kwargs = cfg.get('evaluation', {}).copy()
for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
    eval_kwargs.pop(key, None)
eval_kwargs.update(dict(metric=None, jsonfile_prefix='test', use_image_mask=True))
print('WITH FULL-RUN PADDING:', dataset.evaluate(padded, **eval_kwargs))
