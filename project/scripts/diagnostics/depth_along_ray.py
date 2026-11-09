"""Depth along camera rays: does the model place surfaces too close to the camera on Waymo?

For each camera of a few frames, a ray is cast from the camera center through every
STRIDE-th pixel of the exact model-input image (256x704 after resize/crop), using the
same rots/trans/intrins/post_rots/post_trans the model receives (the inverse of the
projection in project_gt_on_images.py). Each ray is sampled every STEP m through the
200x200x16 grid. Per ray:
  GT first hit   = first sample whose voxel is GT-occupied (not free)
  pred first hit = first sample whose voxel is predicted occupied (not free)
  air filled     = share of samples before the GT first hit that are predicted occupied
Rays are kept only when the GT first hit exists inside the grid and that voxel is in
the evaluation mask (Waymo infov & final_voxel_state, nuScenes mask_camera).
Distances are along the ray from the camera center (ratio = pred / GT).

Sanity checks: GT vs GT (ratio exactly 1, air 0), nuScenes ratio near 1, and the GT
first-hit voxel centers projected forward land back on the ray's own pixel.

CPU only. Usage (alocc env, run from ~/work/ALOcc, project/tools on PYTHONPATH):
  python ~/work/depth_along_ray.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

from class_mapping import MERGED_CLASSES, OTHER, nusc_to_merged, waymo_to_merged

STEP, MAXD, STRIDE = 0.1, 70.0, 4
S = np.arange(STEP / 2, MAXD, STEP, dtype=np.float32)  # sample distances along each ray
H, W = 256, 704
US, VS = np.arange(STRIDE // 2, W, STRIDE), np.arange(STRIDE // 2, H, STRIDE)
LO = np.array([-40, -40, -1], np.float32)
DIMS = np.array([200, 200, 16])
NVOX = 200 * 200 * 16
CLS_NAMES = MERGED_CLASSES[:8] + ['other']
BANDS = [(0, 10), (10, 20), (20, 30), (30, 100)]

W_CFG = '/home/jihoon/work/project/configs/alocc_3d_waymo_zero_shot.py'
W_PRED = '/home/jihoon/work/results/waymo/preds/zero_shot.pkl'
W_OCC = '/home/jihoon/data/occ3d_waymo/validation'
W_FRAMES = [(0, 2)] + [(s, 10) for s in range(20, 201, 20)]  # (scene, kept index k); raw frame = 5k
N_CFG = '/home/jihoon/work/ALOcc/configs/alocc/alocc_3d_r50_256x704_bevdet_preatrain_16f.py'
N_PRED = '/home/jihoon/work/results/batch_nuscenes/preds/baseline.pkl'
N_ANN = '/home/jihoon/work/ALOcc/data/nuscenes/bevdetv2-nuscenes_infos_val.pkl'
N_GT = '/home/jihoon/work/ALOcc/data/nuscenes/gts'
N_FRAMES = [int(i) for i in np.linspace(300, 5700, 10)]
OUT_TXT = '/home/jihoon/work/results/diagnostics/depth_along_ray.txt'
OUT_FIG = '/home/jihoon/work/results/figures/depth_along_ray_s000_f010.png'
FIG_FRAME, FIG_CAM = ('waymo', 0, 10), 'CAM_FRONT'

LINES = []


def log(s=''):
    print(s, flush=True)
    LINES.append(s)


def cam_names(cfg):
    t = cfg.data.test.pipeline[0]['transforms'][0]
    assert t['type'] == 'PrepareImageInputs'
    return t['data_config']['cams']


def model_inputs(data):
    inputs = data['img_inputs'][0]
    inputs = inputs.data if hasattr(inputs, 'data') else inputs
    parts = [t for t in inputs[0]][:6] if isinstance(inputs[0], (list, tuple)) else inputs[:6]
    return [p.numpy().astype(np.float64) for p in parts]


def ray_voxels(R, t, K, P, T):
    """Flat voxel index (NVOX = outside grid) for each ray (row) and sample (column)."""
    uu, vv = np.meshgrid(US, VS)
    pix = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float64)
    q = (pix - T[:2]) @ np.linalg.inv(P[:2, :2]).T          # undo resize/crop
    d = np.concatenate([q, np.ones((len(q), 1))], 1) @ np.linalg.inv(K).T  # camera ray
    d = d @ R.T                                               # camera -> ego
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    pts = t.astype(np.float32) + d.astype(np.float32)[:, None, :] * S[None, :, None]
    ijk = np.floor((pts - LO) / 0.4).astype(np.int32)
    inside = ((ijk >= 0) & (ijk < DIMS)).all(-1)
    flat = ijk[..., 0] * 3200 + ijk[..., 1] * 16 + ijk[..., 2]
    flat[~inside] = NVOX
    return pix, flat


def first_hit(occ_ext, flat):
    hit = occ_ext[flat]
    return hit.any(1), hit.argmax(1), hit


def project(vox, R, t, K, P, T):
    """Forward projection of voxel centers, as in project_gt_on_images.py."""
    ijk = np.stack([vox // 3200, (vox // 16) % 200, vox % 16], 1)
    pts = (ijk + 0.5) * 0.4 + LO
    cam = (pts - t) @ R
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv @ P[:2, :2].T + T[:2], cam[:, 2]


def process_frame(tag, data, names, gt_occ, gt_cls, mask, pred_occ, rec, fig_store):
    imgs, rots, trans, intrins, post_rots, post_trans = model_inputs(data)
    ext = lambda a: np.append(a.ravel(), False)
    g_ext, p_ext, m_ext = ext(gt_occ), ext(pred_occ), ext(mask)
    c_ext = np.append(gt_cls.ravel(), 0)
    for c, name in enumerate(names):
        R, t, K, P, T = rots[c], trans[c], intrins[c][:3, :3], post_rots[c], post_trans[c]
        pix, flat = ray_voxels(R, t, K, P, T)
        ar = np.arange(len(flat))
        gh, gs, _ = first_hit(g_ext, flat)
        ph, ps, phit = first_hit(p_ext, flat)
        gg_h, gg_s, _ = first_hit(g_ext, flat)  # GT used as the prediction (sanity check 1)
        gvox = flat[ar, gs]
        keep = gh & m_ext[gvox]
        gt_d = S[gs].astype(np.float64)
        pred_d = np.where(ph, S[ps], np.inf)
        before = np.cumsum(phit, 1)[ar, np.maximum(gs - 1, 0)] * (gs > 0)
        air = np.where(gs > 0, before / np.maximum(gs, 1), np.nan)
        uv, zc = project(gvox[keep], R, t, K, P, T)
        err = np.linalg.norm(uv - pix[keep], axis=1)
        vox_px = 0.4 * K[0, 0] * P[0, 0] / zc
        k = keep
        rec['tag'] += [tag] * k.sum()
        rec['cam'] += [name] * k.sum()
        rec['gt_d'].append(gt_d[k]); rec['pred_d'].append(pred_d[k]); rec['air'].append(air[k])
        rec['cls'].append(c_ext[gvox[k]]); rec['gg'].append(np.where(gg_h, S[gg_s], np.inf)[k] / gt_d[k])
        rec['err'].append(err); rec['err_norm'].append(err / vox_px)
        rec['n_rays'] += len(flat); rec['n_gt_hit'] += gh.sum()
        if fig_store is not None and name == FIG_CAM:
            shape = (len(VS), len(US))
            img = imgs[2 * c].transpose(1, 2, 0) * [58.395, 57.12, 57.375] + [123.675, 116.28, 103.53]
            fig_store.update(img=np.clip(img / 255, 0, 1), pix=pix[keep], cls=c_ext[gvox[keep]],
                             gt=np.where(gh, gt_d, np.nan).reshape(shape),
                             pred=np.where(ph, pred_d, np.nan).reshape(shape),
                             ratio=np.where(keep, pred_d / gt_d, np.nan).reshape(shape))


def new_rec():
    return dict(tag=[], cam=[], gt_d=[], pred_d=[], air=[], cls=[], gg=[], err=[], err_norm=[], n_rays=0, n_gt_hit=0)


def q(a, p):
    v = np.percentile(np.where(np.isinf(a), 1e3, a), p)
    return 'nohit' if v >= 1e3 else f'{v:.2f}'


def row(label, ratio, air):
    if len(ratio) == 0:
        return
    log(f'{label:<22s} {len(ratio):7d} {q(ratio, 50):>6s} {q(ratio, 25):>6s} {q(ratio, 75):>6s} '
        f'{100 * np.mean(ratio < 0.8):6.1f} {100 * np.mean(ratio > 1.2):6.1f} {100 * np.mean(np.isinf(ratio)):6.1f} '
        f'{np.nanmedian(air):6.3f} {np.nanmean(air):6.3f}')


def summarize(dataset, rec):
    gt_d, pred_d, air = [np.concatenate(rec[k]) for k in ('gt_d', 'pred_d', 'air')]
    cls, gg = np.concatenate(rec['cls']), np.concatenate(rec['gg'])
    err, err_norm = np.concatenate(rec['err']), np.concatenate(rec['err_norm'])
    cam, tag = np.array(rec['cam']), np.array(rec['tag'])
    ratio = pred_d / gt_d
    log(f'===== {dataset}: {len(set(tag))} frames, {rec["n_rays"]} rays cast, {rec["n_gt_hit"]} with a GT hit in the grid, '
        f'{len(ratio)} kept (GT first hit inside the eval mask)')
    log('ratio = pred first-hit distance / GT first-hit distance (along the ray); <0.8 = pred >20% closer; '
        'nohit% = pred has no occupied voxel on the ray inside the grid (ratio counted as +inf)')
    log('air = share of the ray before the GT first hit that the pred fills (occupied)')
    hdr = f'{"group":<22s} {"n":>7s} {"median":>6s} {"q25":>6s} {"q75":>6s} {"<0.8%":>6s} {">1.2%":>6s} {"nohit%":>6s} {"airMed":>6s} {"airMean":>6s}'
    log(hdr)
    row('ALL', ratio, air)
    log('-- by GT class of the first hit (merged classes)')
    for i, n in enumerate(CLS_NAMES):
        sel = (cls == (OTHER if n == 'other' else i))
        row(n, ratio[sel], air[sel])
    log('-- by camera')
    for n in dict.fromkeys(rec['cam']):
        row(n, ratio[cam == n], air[cam == n])
    log('-- by GT first-hit distance (m)')
    for a, b in BANDS:
        sel = (gt_d >= a) & (gt_d < b)
        row(f'{a}-{b}', ratio[sel], air[sel])
    log('-- by frame')
    for n in dict.fromkeys(rec['tag']):
        row(n, ratio[tag == n], air[tag == n])
    log(f'median |pred - GT| distance on rays with a pred hit: {np.median(np.abs(pred_d - gt_d)[np.isfinite(pred_d)]):.2f} m; '
        f'median GT first-hit distance {np.median(gt_d):.2f} m')
    log(f'SANITY 1 (GT as pred): max |ratio - 1| = {np.max(np.abs(gg - 1)):.2e} over {len(gg)} rays')
    log(f'SANITY 3 (GT first-hit voxel center projected forward vs the ray pixel): median {np.median(err):.2f} px, '
        f'95th pct {np.percentile(err, 95):.2f} px; in units of the projected voxel width: median {np.median(err_norm):.2f}, '
        f'95th pct {np.percentile(err_norm, 95):.2f}, max {err_norm.max():.2f} (<= ~0.87 expected for a 3D voxel)')
    log()
    return ratio


def run_nuscenes():
    infos = sorted(pickle.load(open(N_ANN, 'rb'))['infos'], key=lambda e: e['timestamp'])
    preds = {}
    for p in mmcv.load(N_PRED):
        if p['index'] in N_FRAMES:
            preds.setdefault(p['index'], p['pred_occupancy'])
    cfg = Config.fromfile(N_CFG)
    ds = build_dataset(cfg.data.test)
    names = cam_names(cfg)
    rec = new_rec()
    for i in N_FRAMES:
        info = infos[i]
        assert ds.data_infos[i]['token'] == info['token'], i
        g = np.load(f'{N_GT}/{info["scene_name"]}/{info["token"]}/labels.npz')
        lab = g['semantics']
        process_frame(f'nus {info["scene_name"]} i{i}', ds[i], names, lab != 17, nusc_to_merged(lab),
                      g['mask_camera'].astype(bool), preds[i] != 17, rec, None)
        print('done nuScenes', i, flush=True)
    return summarize('nuScenes (baseline predictions)', rec)


def run_waymo(fig_store):
    want = {(s, 5 * k) for s, k in W_FRAMES}
    preds = {}
    for p in mmcv.load(W_PRED):
        if (p['scene_idx'], p['frame_idx']) in want:
            preds[(p['scene_idx'], p['frame_idx'])] = p['pred_occupancy']
    assert set(preds) == want, want - set(preds)
    rec = new_rec()
    for s, k in W_FRAMES:
        cfg = Config.fromfile(W_CFG)
        cfg.data.test.scene_range = [s, s + 1]
        ds = build_dataset(cfg.data.test)
        f = 5 * k
        g = np.load(f'{W_OCC}/{s:03d}/{f:03d}_04.npz')
        lab = g['voxel_label']
        mask = g['infov'].astype(bool) & g['final_voxel_state'].astype(bool)
        store = fig_store if ('waymo', s, f) == FIG_FRAME else None
        process_frame(f'waymo s{s:03d} f{f:03d}', ds[k], cam_names(cfg), lab != 23, waymo_to_merged(lab),
                      mask, preds[(s, f)] != 17, rec, store)
        print('done Waymo', s, f, flush=True)
    return summarize('Waymo (zero_shot predictions)', rec)


def make_figure(st):
    colors = np.array(['blue', 'red', 'orange', 'yellow', 'gray', 'magenta', 'green', 'saddlebrown', 'black'])
    ext = (0, W, H, 0)
    fig, ax = plt.subplots(5, 1, figsize=(12, 22))
    ax[0].imshow(st['img']); ax[0].set_title('Waymo scene 0 frame 10, FRONT: model input image (256x704)')
    for a, key, title in ((ax[1], 'gt', 'GT first-hit distance along ray (m), all rays hitting GT in the grid'),
                          (ax[2], 'pred', 'pred first-hit distance along ray (m)')):
        im = a.imshow(st[key], cmap='turbo', vmin=0, vmax=40, extent=ext, interpolation='nearest')
        a.set_title(title); plt.colorbar(im, ax=a, fraction=0.025)
    im = ax[3].imshow(st['ratio'], cmap='RdBu', vmin=0, vmax=2, extent=ext, interpolation='nearest')
    ax[3].set_title('pred / GT first-hit distance (kept rays: GT hit inside eval mask); red < 1 = pred too close')
    plt.colorbar(im, ax=ax[3], fraction=0.025)
    cls = np.where(st['cls'] == OTHER, 8, st['cls'])
    ax[4].imshow(st['img'])
    ax[4].scatter(st['pix'][:, 0], st['pix'][:, 1], s=3, c=colors[cls], alpha=0.7)
    ax[4].set_title('sanity: GT class of the first hit on kept rays (compare gt_on_images_s000_f010.png slot 2)')
    for a in ax:
        a.set_xlim(0, W); a.set_ylim(H, 0)
    fig.suptitle('blue vehicle, red ped, orange 2-wheel, yellow cone, gray road, magenta sidewalk, green veg, brown manmade, black other')
    plt.tight_layout(); plt.savefig(OUT_FIG, dpi=60)


if __name__ == '__main__':
    log(f'# depth_along_ray.py: step {STEP} m, max {MAXD} m, every {STRIDE}th pixel of the 256x704 model input')
    log(f'# Waymo frames (scene, raw frame): {[(s, 5 * k) for s, k in W_FRAMES]}; nuScenes indices: {N_FRAMES}')
    log()
    r_nus = run_nuscenes()
    fig_store = {}
    r_way = run_waymo(fig_store)
    log(f'SANITY 2: nuScenes median ratio {q(r_nus, 50)} (Waymo {q(r_way, 50)})')
    make_figure(fig_store)
    open(OUT_TXT, 'w').write('\n'.join(LINES) + '\n')
    print('saved', OUT_TXT, OUT_FIG)
