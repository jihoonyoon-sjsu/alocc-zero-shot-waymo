"""Join chunk predictions (from make_chunk_configs.py runs) into one result list
for the full split, then score it with the dataset's own evaluation, the same
call ALOcc's tools/test.py makes after a full run.

Each chunk's 'index' counts from 0 inside that chunk, so the chunk offset is added.
ALOcc's test loader pads the last batch by repeating frames (17 extra results per
run), so repeated indices are checked to be identical and kept once. Note that a
full ALOcc run scores those padded frames twice; this merge scores each frame once.

Usage (run from the ALOcc folder):
  python merge_chunks.py CHUNKS_JSON CHUNK_PRED_PATTERN OUT_PKL
  CHUNK_PRED_PATTERN contains {i}, e.g. preds/baseline_chunk{i}.pkl
"""
import json
import sys

import mmcv
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

chunks_json, pattern, out_pkl = sys.argv[1:4]
meta = json.load(open(chunks_json))

merged = []
for c in meta['chunks']:
    preds = mmcv.load(pattern.format(i=c['chunk']))
    kept = {}
    for p in preds:
        if p['index'] in kept:
            assert np.array_equal(kept[p['index']]['pred_occupancy'], p['pred_occupancy']), (c, p['index'])
        else:
            kept[p['index']] = p
    assert sorted(kept) == list(range(c['length'])), (c, len(kept))
    print(f"chunk {c['chunk']}: {len(preds)} results, {len(kept)} unique frames")
    for i in sorted(kept):
        kept[i]['index'] = i + c['offset']
        merged.append(kept[i])
    del preds
assert sorted(p['index'] for p in merged) == list(range(meta['total']))
mmcv.dump(merged, out_pkl)
print('merged', len(merged), '->', out_pkl)

cfg = Config.fromfile(meta['config'])
cfg.data.test.test_mode = True
dataset = build_dataset(cfg.data.test)
assert len(dataset) == meta['total']
eval_kwargs = cfg.get('evaluation', {}).copy()
for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best', 'rule']:
    eval_kwargs.pop(key, None)
eval_kwargs.update(dict(metric=None, jsonfile_prefix='test', use_image_mask=True))
print(dataset.evaluate(merged, **eval_kwargs))
