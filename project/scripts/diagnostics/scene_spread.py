import json
import numpy as np
r = json.load(open('/home/jihoon/work/results/waymo/zero_shot.score.json'))
s = r['scene_occupied_iou']
v = np.array(list(s.values()))
print('scenes', len(v), 'min %.1f p10 %.1f median %.1f p90 %.1f max %.1f' % (v.min(), *np.percentile(v, [10, 50, 90]), v.max()))
print('lowest:', sorted(s.items(), key=lambda kv: kv[1])[:8])
print('highest:', sorted(s.items(), key=lambda kv: -kv[1])[:5])
c = np.array(r['confusion_rows_gt_cols_pred']['counts']); L = r['confusion_rows_gt_cols_pred']['labels']
print('GT share of masked voxels:', {L[i]: round(100 * c[i].sum() / c.sum(), 1) for i in range(len(L))})
print('pred share:', {L[i]: round(100 * c[:, i].sum() / c.sum(), 1) for i in range(len(L))})
