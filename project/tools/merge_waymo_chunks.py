"""Join zero-shot Waymo chunk predictions (one run per scene range) into one file.

Each chunk's 'index' counts frames inside that chunk, in the order used by
WaymoOccZeroShot (sorted by image_idx, every 5th frame). This maps each index to
its (scene_idx, frame_idx), keeps each frame once (the test loader repeats a few
frames to fill its last batch; repeats are checked identical), and saves a list of
dicts with scene_idx, frame_idx, pred_occupancy.

Usage (alocc env, project/tools on PYTHONPATH):
  python merge_waymo_chunks.py ANN_FILE "0:51,51:101,..." CHUNK_PRED_PATTERN OUT_PKL
  CHUNK_PRED_PATTERN contains {i} (chunk number), e.g. preds/zero_shot_chunk{i}.pkl
"""
import pickle
import sys

import mmcv
import numpy as np

from waymo_zero_shot import keep_frame, scene_frame

ann_file, ranges, pattern, out_pkl = sys.argv[1:5]
ranges = [tuple(int(v) for v in r.split(':')) for r in ranges.split(',')]
infos = sorted(pickle.load(open(ann_file, 'rb')), key=lambda i: i['image']['image_idx'])

merged = []
for c, scene_range in enumerate(ranges):
    frames = [scene_frame(i) for i in infos if keep_frame(i, 5, scene_range)]
    kept = {}
    for p in mmcv.load(pattern.format(i=c)):
        if p['index'] in kept:
            assert np.array_equal(kept[p['index']], p['pred_occupancy']), (c, p['index'])
        else:
            kept[p['index']] = p['pred_occupancy']
    assert sorted(kept) == list(range(len(frames))), (c, len(kept), len(frames))
    for i, (scene, frame) in enumerate(frames):
        merged.append(dict(scene_idx=scene, frame_idx=frame, pred_occupancy=kept[i]))
    print(f'chunk {c} scenes {scene_range}: {len(frames)} frames')
assert len({(m['scene_idx'], m['frame_idx']) for m in merged}) == len(merged)
mmcv.dump(merged, out_pkl)
print('merged', len(merged), '->', out_pkl)
