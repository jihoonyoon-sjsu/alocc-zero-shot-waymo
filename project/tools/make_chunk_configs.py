"""Split a test run into N chunks so each chunk saves its predictions on its own.

The test infos are sorted by timestamp (as ALOcc's NuScenesDataset does) and cut
at scene boundaries, so every scene stays whole and the model's temporal history
sees the same frames as in a single full run. Use merge_chunks.py to join the
chunk predictions back into one result list for the full split.

Usage: python make_chunk_configs.py CONFIG OUT_DIR NAME [N_CHUNKS=3]
Writes OUT_DIR/{NAME}_chunk{i}.pkl, OUT_DIR/{NAME}_chunk{i}.py and
OUT_DIR/{NAME}_chunks.json (offset and length of each chunk in the full split).
"""
import json
import os
import pickle
import sys

from mmcv import Config


def set_ann_file(node, old_path, new_path):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == 'ann_file' and v == old_path:
                node[k] = new_path
            else:
                set_ann_file(v, old_path, new_path)
    elif isinstance(node, list):
        for v in node:
            set_ann_file(v, old_path, new_path)


cfg_path, out_dir, name = sys.argv[1:4]
n_chunks = int(sys.argv[4]) if len(sys.argv) > 4 else 3
os.makedirs(out_dir, exist_ok=True)

cfg = Config.fromfile(cfg_path)
ann_file = cfg.data.test.ann_file
with open(ann_file, 'rb') as f:
    data = pickle.load(f)
infos = sorted(data['infos'], key=lambda e: e['timestamp'])

# Scene start positions, then pick the scene start closest to each even cut.
starts = [i for i in range(len(infos)) if i == 0 or infos[i]['scene_name'] != infos[i - 1]['scene_name']]
cuts = [0]
for c in range(1, n_chunks):
    target = round(c * len(infos) / n_chunks)
    cuts.append(min(starts, key=lambda s: abs(s - target)))
cuts.append(len(infos))
assert cuts == sorted(set(cuts)), cuts

chunks = []
for i in range(n_chunks):
    part = dict(data)
    part['infos'] = infos[cuts[i]:cuts[i + 1]]
    pkl_path = os.path.abspath(os.path.join(out_dir, f'{name}_chunk{i}.pkl'))
    with open(pkl_path, 'wb') as f:
        pickle.dump(part, f)
    chunk_cfg = Config.fromfile(cfg_path)
    set_ann_file(chunk_cfg._cfg_dict, ann_file, pkl_path)
    assert chunk_cfg.data.test.ann_file == pkl_path
    chunk_cfg.dump(os.path.join(out_dir, f'{name}_chunk{i}.py'))
    chunks.append(dict(chunk=i, offset=cuts[i], length=cuts[i + 1] - cuts[i]))
    print(f'{name} chunk {i}: frames {cuts[i]}-{cuts[i + 1] - 1} ({cuts[i + 1] - cuts[i]})')

with open(os.path.join(out_dir, f'{name}_chunks.json'), 'w') as f:
    json.dump(dict(config=os.path.abspath(cfg_path), ann_file=ann_file, total=len(infos), chunks=chunks), f, indent=2)
