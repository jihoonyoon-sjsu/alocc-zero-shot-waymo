# Project folder map

Everything we wrote lives here. ALOcc's own code (the rest of the repo) is unmodified. Nothing was trained: every result comes from running the released ALOcc-3D nuScenes checkpoint and scoring its predictions.

Folders are listed in the order the pipeline uses them.

## `tools/`: core code

These are the Python modules the runs and scorers import. They stay in one flat folder because they import each other by name (`tools/` must be on `PYTHONPATH`).

- `waymo_zero_shot.py`: the Waymo adapter. Subsamples Waymo to 2 Hz, recomputes ego motion, and converts the camera calibration to the nuScenes convention so the nuScenes checkpoint can read it.
- `extract_waymo_images.py`: pulls the original JPEGs out of the Waymo v2 parquet files (run in the `waymo` env).
- `class_mapping.py`: maps nuScenes and Waymo classes onto the 8 shared classes.
- `make_cam_dropout_config.py`: writes a config with one camera removed.
- `make_extrinsic_noise_config.py`: writes a config with BEVFormer-style calibration noise.
- `make_chunk_configs.py`, `merge_chunks.py`, `merge_waymo_chunks.py`: split long runs into chunks at scene boundaries, then merge them and remove ALOcc's padded duplicate frames.
- `occ_scoring.py`: shared scoring core (occupied IoU and mIoU, split by distance and camera direction).
- `score_nuscenes_merged.py`, `score_waymo.py`: score a prediction file on each dataset.
- `run_alocc_test.sh`: wrapper around ALOcc's test script.

## `configs/`

- `alocc_3d_waymo_zero_shot.py`: the Waymo config (5 cameras, image resize and crop, adapter dataset). It is based on ALOcc's nuScenes config.

## `scripts/`: how the runs were driven

These are copies of the helper scripts that ran in WSL. Paths inside them are machine-specific (`~/work/...`), so read them as a record of exactly what was run, or adjust the paths before reusing them.

- `runs/`: GPU runs. `batch_nuscenes.sh` (the six camera dropouts, unchunked), `batch_chunked.sh` (nuScenes baseline and noise levels 1 to 4, in chunks), `make_noise_configs.sh`, `waymo_smoke.sh` (40-frame trial), `batch_waymo.sh` (full Waymo zero-shot and history-reset runs).
- `scoring/`: `score_all.sh` (nuScenes baseline and both Waymo runs), `score_robustness.sh` (dropout and noise runs), `list_miou.sh` (collects official mIoU from the logs).
- `checks/`: pipeline validation. Index alignment, camera slot order, adapter output, smoke test, chunk merge, history reset, image/calibration pairing on all 202 scenes, and ground truth projected onto the model's input images.
- `diagnostics/`: failure analysis. Height profiles (Waymo and nuScenes), false-occupied by distance and height, class shares, mask contents, top-down visualization, and depth along camera rays. `rerun_diagnostics.sh` reruns them all.

## `results/`: outputs

- `nuscenes/`: baseline reproduction (`baseline.merged.log`, 45.46 mIoU) and its 8-class score file.
  - `camera_dropout/`: official logs for the six one-camera-removed runs.
  - `extrinsic_noise/`: official logs for noise levels 1 to 4, and the exact noise drawn per camera (`noise_level*.json`).
  - `merged8_scores/`: 8-class scores for every dropout and noise run, split by distance and camera direction.
- `waymo/`: logs and score files for the zero-shot and history-reset runs (8,069 frames each). Each score file includes per-scene occupied IoU.
- `diagnostics/`: text outputs of every check and diagnostic script, one file per script.

## Not included

- Datasets (nuScenes, Waymo, Occ3D labels) and the checkpoint: download them from their official sources.
- Prediction files (4 to 5 GB each).
- Figures containing Waymo camera images (Waymo dataset license).

## Running

Everything ran in WSL2 Ubuntu 22.04 with conda envs `alocc` (Python 3.8, PyTorch 2.0.1, mmcv 1.7.2) and `waymo` (Python 3.10, waymo-open-dataset). Run from the ALOcc root, with `project/tools` on `PYTHONPATH`. Example:

```bash
bash tools/dist_test.sh project/configs/alocc_3d_waymo_zero_shot.py ckpts/alocc_3d_r50_256x704_bevdet_preatrain_16f.pth 1 --out zero_shot.pkl
```
