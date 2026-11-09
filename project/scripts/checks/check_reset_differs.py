"""Check the history-reset Waymo run differs from the normal run (chunk 0 only)."""
import mmcv
import numpy as np

W = '/home/jihoon/work/results/waymo/preds'
a = {p['index']: p['pred_occupancy'] for p in mmcv.load(f'{W}/zero_shot_chunk0.pkl')}
b = {p['index']: p['pred_occupancy'] for p in mmcv.load(f'{W}/history_reset_chunk0.pkl')}
same = sum(np.array_equal(a[i], b[i]) for i in a)
diff = np.mean([(a[i] != b[i]).mean() for i in a])
print(f'frames {len(a)}, identical frames {same}, mean fraction of voxels that differ {100 * diff:.2f}%')
