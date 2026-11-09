"""Score ALOcc nuScenes predictions on the same 8 merged classes as Waymo, so the
two datasets can be compared on equal terms.

Uses Occ3D-nuScenes mask_camera (ALOcc's default evaluation mask). Reports the
same numbers as score_waymo.py: occupied IoU, merged-class IoU and mIoU, split by
distance band and camera sector (nearest of the six cameras by azimuth).

Usage (alocc env, project/tools on PYTHONPATH, run from the ALOcc folder):
  python score_nuscenes_merged.py PRED_PKL OUT_JSON [ANN_FILE] [GT_DIR]
  PRED_PKL is a merged ALOcc test output (list of dicts with index, pred_occupancy);
  index follows ALOcc's order (infos sorted by timestamp).
"""
import json
import pickle
import sys
from multiprocessing import Pool

import mmcv
import numpy as np
from pyquaternion import Quaternion

from class_mapping import nusc_to_merged
from occ_scoring import BANDS, N, frame_hists, report, sector_map

ann_file = sys.argv[3] if len(sys.argv) > 3 else 'data/nuscenes/bevdetv2-nuscenes_infos_val.pkl'
gt_dir = sys.argv[4] if len(sys.argv) > 4 else 'data/nuscenes/gts'
CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

INFOS = sorted(pickle.load(open(ann_file, 'rb'))['infos'], key=lambda e: e['timestamp'])
ITEMS = []  # (index, pred); filled before the worker pool forks


def cam_yaw(cam):
    axis = Quaternion(cam['sensor2ego_rotation']).rotation_matrix[:, 2]  # optical axis in ego frame
    return float(np.degrees(np.arctan2(axis[1], axis[0])))


YAWS = [cam_yaw(INFOS[0]['cams'][c]) for c in CAMS]
SECTORS = sector_map(YAWS)


def score_frame(i):
    index, pred = ITEMS[i]
    info = INFOS[index]
    gt = np.load(f'{gt_dir}/{info["scene_name"]}/{info["token"]}/labels.npz')
    mask = gt['mask_camera'].astype(bool)
    return frame_hists(nusc_to_merged(gt['semantics']), nusc_to_merged(pred), mask, SECTORS, len(CAMS))


if __name__ == '__main__':
    pred_pkl, out_json = sys.argv[1:3]
    print('camera yaws (deg):', dict(zip(CAMS, np.round(YAWS, 1))))
    kept = {}
    for p in mmcv.load(pred_pkl):
        kept.setdefault(p['index'], p['pred_occupancy'])
    assert sorted(kept) == list(range(len(INFOS))), (len(kept), len(INFOS))
    ITEMS.extend(sorted(kept.items()))
    del kept
    total = np.zeros((N, N), np.int64)
    band = np.zeros((len(BANDS), N, N), np.int64)
    sec = np.zeros((len(CAMS), N, N), np.int64)
    with Pool(6) as pool:
        for t, b, s in pool.imap_unordered(score_frame, range(len(ITEMS)), chunksize=16):
            total += t
            band += b
            sec += s
    result = report(pred_pkl, len(ITEMS), total, band, sec, CAMS)
    json.dump(result, open(out_json, 'w'), indent=1)
