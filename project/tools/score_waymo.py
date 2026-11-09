"""Score zero-shot Waymo predictions against Occ3D-Waymo ground truth.

Uses the same mask as ALOcc's Waymo evaluation (infov & final_voxel_state) and
the 8 merged classes from class_mapping.py. Reports occupied IoU (occupied vs
free, ignoring class), IoU per merged class and their mean (mIoU, free not
included), the same split by distance band and camera sector, and occupied IoU
per scene. Before scoring, checks the prediction file covers every expected
(scene, frame) exactly once with valid shapes and labels.

Usage (alocc env, project/tools on PYTHONPATH):
  python score_waymo.py PRED_PKL OUT_JSON [ANN_FILE] [OCC_DIR]
  PRED_PKL is a merge_waymo_chunks.py output (scene_idx, frame_idx, pred_occupancy).
"""
import json
import pickle
import sys
from multiprocessing import Pool

import mmcv
import numpy as np

from class_mapping import nusc_to_merged, waymo_to_merged
from occ_scoring import BANDS, N, frame_hists, occupied_iou, report, sector_map
from waymo_zero_shot import keep_frame, scene_frame

ann_file = sys.argv[3] if len(sys.argv) > 3 else '/home/jihoon/data/waymo_alocc/waymo_infos_val_jpg.pkl'
occ_dir = sys.argv[4] if len(sys.argv) > 4 else '/home/jihoon/data/occ3d_waymo/validation'
# Waymo camera yaws (verified from the extrinsics: 0, +45, +90, -45, -90 degrees).
SECTOR_NAMES = ['FRONT', 'FRONT_LEFT', 'SIDE_LEFT', 'FRONT_RIGHT', 'SIDE_RIGHT']
SECTORS = sector_map([0, 45, 90, -45, -90])

ITEMS = []  # (scene, frame, pred); filled before the worker pool forks, so workers share it


def score_frame(i):
    scene, frame, pred = ITEMS[i]
    gt = np.load(f'{occ_dir}/{scene:03d}/{frame:03d}_04.npz')
    mask = gt['infov'].astype(bool) & gt['final_voxel_state'].astype(bool)
    hists = frame_hists(waymo_to_merged(gt['voxel_label']), nusc_to_merged(pred), mask, SECTORS, len(SECTOR_NAMES))
    return (scene,) + hists


def validate(items):
    infos = pickle.load(open(ann_file, 'rb'))
    expected = {scene_frame(i) for i in infos if keep_frame(i, 5)}
    got = [(s, f) for s, f, _ in items]
    assert len(got) == len(set(got)), 'duplicate (scene, frame) entries'
    assert set(got) == expected, f'missing {len(expected - set(got))}, unexpected {len(set(got) - expected)}'
    for _, _, p in items:
        assert p.shape == (200, 200, 16) and p.max() <= 17, (p.shape, p.max())
    print(f'validated: {len(got)} unique frames = every expected frame, shapes and labels OK')


if __name__ == '__main__':
    pred_pkl, out_json = sys.argv[1:3]
    preds = mmcv.load(pred_pkl)
    ITEMS.extend((p['scene_idx'], p['frame_idx'], p['pred_occupancy']) for p in preds)
    del preds
    validate(ITEMS)
    total = np.zeros((N, N), np.int64)
    band = np.zeros((len(BANDS), N, N), np.int64)
    sec = np.zeros((len(SECTOR_NAMES), N, N), np.int64)
    per_scene = {}
    with Pool(6) as pool:
        for scene, t, b, s in pool.imap_unordered(score_frame, range(len(ITEMS)), chunksize=16):
            total += t
            band += b
            sec += s
            per_scene[scene] = per_scene.get(scene, 0) + t
    result = report(pred_pkl, len(ITEMS), total, band, sec, SECTOR_NAMES)
    result['scene_occupied_iou'] = {int(s): 100 * occupied_iou(h) for s, h in sorted(per_scene.items())}
    json.dump(result, open(out_json, 'w'), indent=1)
    print(f'scene 0 occupied IoU {result["scene_occupied_iou"].get(0, float("nan")):.2f}')
