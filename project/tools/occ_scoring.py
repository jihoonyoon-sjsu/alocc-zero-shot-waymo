"""Shared scoring core for the 8 merged classes (see class_mapping.py).

Every frame adds to confusion matrices (rows = GT, cols = prediction, over the
merged classes plus an "other" bucket) for: all voxels, each distance band
(horizontal distance from the ego origin) and each camera sector (the camera
whose viewing direction is closest in azimuth). Only voxels inside the given
mask are counted.
"""
import numpy as np

from class_mapping import MERGED_CLASSES, OTHER

FREE = MERGED_CLASSES.index('free')
N = len(MERGED_CLASSES) + 1  # merged classes plus "other" (last)
BANDS = ['0-10m', '10-20m', '20-30m', '30-40m', '40m+']

# Voxel centres of the shared 200x200x16 grid ([-40,40]^2 x [-1,5.4], 0.4 m).
_c = (np.arange(200) + 0.5) * 0.4 - 40
_X, _Y = np.meshgrid(_c, _c, indexing='ij')
DIST_BAND = np.minimum((np.hypot(_X, _Y) // 10).astype(np.int64), len(BANDS) - 1)[:, :, None].repeat(16, 2)
AZIMUTH = np.degrees(np.arctan2(_Y, _X))[:, :, None].repeat(16, 2)  # 0 = forward, +90 = left


def sector_map(cam_yaws):
    """Index of the camera (in cam_yaws order) whose yaw is closest to each voxel's azimuth."""
    diff = np.abs((AZIMUTH[..., None] - np.array(cam_yaws) + 180) % 360 - 180)
    return np.argmin(diff, -1)


def to_bins(merged):
    out = merged.astype(np.int64)
    out[merged == OTHER] = N - 1
    return out


def frame_hists(gt_merged, pred_merged, mask, sectors, n_sectors):
    """Confusion matrices for one frame: total (N,N), per band (B,N,N), per sector (S,N,N)."""
    g = to_bins(gt_merged[mask])
    p = to_bins(pred_merged[mask])
    cell = g * N + p
    total = np.bincount(cell, minlength=N * N).reshape(N, N)
    band = np.bincount(DIST_BAND[mask] * N * N + cell, minlength=len(BANDS) * N * N).reshape(len(BANDS), N, N)
    sec = np.bincount(sectors[mask] * N * N + cell, minlength=n_sectors * N * N).reshape(n_sectors, N, N)
    return total, band, sec


def iou(hist, c):
    return hist[c, c] / max(hist[c, :].sum() + hist[:, c].sum() - hist[c, c], 1)


def occupied_iou(hist):
    occ = [c for c in range(N) if c != FREE]
    return hist[np.ix_(occ, occ)].sum() / max(hist.sum() - hist[FREE, FREE], 1)


def summarize(hist):
    class_iou = {name: 100 * iou(hist, c) for c, name in enumerate(MERGED_CLASSES) if c != FREE}
    return dict(occupied_iou=100 * occupied_iou(hist),
                miou=float(np.mean(list(class_iou.values()))),
                class_iou=class_iou,
                voxels=int(hist.sum()))


def report(name, frames, total, band, sec, sector_names):
    result = dict(pred_file=name, frames=frames, **summarize(total),
                  distance=dict(zip(BANDS, map(summarize, band))),
                  camera_sector=dict(zip(sector_names, map(summarize, sec))),
                  confusion_rows_gt_cols_pred=dict(labels=MERGED_CLASSES + ['other'], counts=total.tolist()))
    print(f'{name}: {frames} frames')
    print(f'occupied IoU {result["occupied_iou"]:.2f}  mIoU (8 merged) {result["miou"]:.2f}')
    for k, v in result['class_iou'].items():
        print(f'  {k:17s} {v:6.2f}')
    for title, part in [('distance', result['distance']), ('camera sector', result['camera_sector'])]:
        print(f'by {title}:')
        for k, v in part.items():
            print(f'  {k:16s} occupied IoU {v["occupied_iou"]:6.2f}  mIoU {v["miou"]:6.2f}  voxels {v["voxels"]}')
    return result
