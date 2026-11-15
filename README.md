# Same Model, New Car?

Diagnosing zero-shot transfer of a pretrained 3D semantic occupancy model (ALOcc-3D) from nuScenes to Waymo. We run the released nuScenes checkpoint on Occ3D-Waymo with no retraining, measure the drop, and diagnose why.

Video demo: VIDEO_LINK_TO_BE_ADDED

CMPE 249 – Intelligent Autonomous Systems, SJSU. Project Track: Option 1 – Apply and Evaluate Pretrained Models (No Training).

Team: Simisola Winjobi, JiHoon Yoon, Tara Fathinejad

## Built on ALOcc

This repository includes the code of **ALOcc** (Chen et al., ICCV 2025), copied unmodified from [cdb342/ALOcc](https://github.com/cdb342/ALOcc) at commit `1bfd7de`, under the Apache 2.0 license (see `LICENSE`). ALOcc's original README is in [README_ALOcc.md](README_ALOcc.md).

```bibtex
@InProceedings{chen2025alocc,
    author    = {Chen, Dubing and Fang, Jin and Han, Wencheng and Cheng, Xinjing and Yin, Junbo and Xu, Chenzhong and Khan, Fahad Shahbaz and Shen, Jianbing},
    title     = {Alocc: adaptive lifting-based 3d semantic occupancy and cost volume-based flow prediction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
}
```

## Repository layout

- `project/`: **our work** (Waymo adapter, configs, scoring, run scripts, results). See [project/README.md](project/README.md) for a map of every folder.
  - `project/robustness/`: image corruption and crop-shift experiments.
- `ros2/`: ROS 2 packages (messages, Waymo replay, RViz). See [ros2/README_ROS2.md](ros2/README_ROS2.md).
  - `ros2/live_model/`: the live ALOcc-3D model inside ROS 2, with latency and resource results. See [ros2/live_model/README.md](ros2/live_model/README.md).
- Everything else: ALOcc's code, unmodified.

## Main results

Nothing was trained. All numbers come from the released ALOcc-3D nuScenes checkpoint, scored on 8 merged classes.

| | nuScenes val (6,019 frames) | Waymo val, zero-shot (8,069 frames) |
|---|---|---|
| Occupied IoU | 75.30 | 22.48 |
| mIoU (8 classes) | 50.25 | 9.56 |

- nuScenes reproduction: 45.46 official mIoU (published 45.5).
- Removing the rear camera on nuScenes costs only about 5 mIoU (45.46 to 40.24), and resetting the 16-frame history on Waymo changes scores by about 1 point.
- On Waymo, the model over-fills free space and raises the road surface by about 0.8 m, while the top-down layout stays roughly right.
- Shifting the image crop up to move the Waymo horizon toward the nuScenes position made results slightly worse (shift 40 px, which matches nuScenes: 8.81 mIoU; shift 50 px: 8.70; original: 9.56).
- Live ROS 2 run on an RTX 4070 Ti: 812 ms end-to-end per frame (1.23 FPS), identical to the offline predictions.

Full tables are in `project/results/` and in the report.

## Setup

Tested on WSL2 Ubuntu 22.04 with one RTX 4070 Ti (12 GB).

1. **ALOcc environment** (`alocc`, Python 3.8): follow the installation steps in [README_ALOcc.md](README_ALOcc.md) (PyTorch 2.0.1 + CUDA 11.8, mmcv 1.x built with ops, mmdet 2.28.2, mmsegmentation 0.30.0, then `pip install -v -e .`).
2. **Waymo environment** (`waymo`, Python 3.10): waymo-open-dataset 1.6.7 and pyarrow (used only to extract images and check calibration).
3. **Checkpoint:** download `alocc_3d_r50_256x704_bevdet_preatrain_16f.pth` from the [ALOcc Hugging Face Hub](https://huggingface.co/Dobbin/OccStudio/blob/main/alocc_3d_r50_256x704_bevdet_preatrain_16f.pth) into `ckpts/`.
4. **Data:**
   - nuScenes and Occ3D-nuScenes: see [README_ALOcc.md](README_ALOcc.md) ([nuScenes](https://www.nuscenes.org/download), [Occ3D](https://tsinghua-mars-lab.github.io/Occ3D/)).
   - Waymo Open Dataset v2.0.1 validation (camera image, camera calibration and pose components only) from [waymo.com/open](https://waymo.com/open/), plus the Occ3D-Waymo validation labels from the [Occ3D page](https://tsinghua-mars-lab.github.io/Occ3D/).

## How to run (evaluation only)

Run from the repository root with `project/tools` on `PYTHONPATH`, and set two data folders:

```bash
export PYTHONPATH=$PWD/project/tools:$PYTHONPATH
export WAYMO_V2_VAL_DIR=/path/to/waymo_v2/validation   # Waymo v2.0.1 validation, contains camera_image/*.parquet (202 segments)
export OCC3D_WAYMO_ROOT=/path/to/occ3d_waymo   # contains waymo_infos_val.pkl, cam_infos_vali.pkl, validation/
export WAYMO_ROOT=/path/to/waymo_alocc         # output folder for extracted images
```

1. Extract Waymo images (in the `waymo` env). This also writes `$WAYMO_ROOT/waymo_infos_val_jpg.pkl`:

   ```bash
   python project/tools/extract_waymo_images.py $WAYMO_V2_VAL_DIR $OCC3D_WAYMO_ROOT $WAYMO_ROOT
   ```

2. nuScenes baseline:

   ```bash
   bash tools/dist_test.sh configs/alocc/alocc_3d_r50_256x704_bevdet_preatrain_16f.py ckpts/alocc_3d_r50_256x704_bevdet_preatrain_16f.pth 1 --out nus_baseline.pkl
   ```

3. Waymo zero-shot:

   ```bash
   bash tools/dist_test.sh project/configs/alocc_3d_waymo_portable.py ckpts/alocc_3d_r50_256x704_bevdet_preatrain_16f.pth 1 --out zero_shot.pkl
   ```

   `alocc_3d_waymo_portable.py` is identical to the `alocc_3d_waymo_zero_shot.py` config we ran, except that it reads the data folders from `WAYMO_ROOT` and `OCC3D_WAYMO_ROOT` instead of our machine's paths.

   For the history-reset ablation, add `--cfg-options data.test.sequences_split_num=all`.

4. Score on 8 merged classes:

   ```bash
   python project/tools/score_waymo.py zero_shot.pkl zero_shot.score.json $WAYMO_ROOT/waymo_infos_val_jpg.pkl $OCC3D_WAYMO_ROOT/validation
   python project/tools/score_nuscenes_merged.py nus_baseline.pkl nus_baseline.score.json
   ```

   The Waymo scorer's default paths are machine-specific, so always pass the info file and label folder as above.

5. Camera dropout and extrinsic noise configs: `project/tools/make_cam_dropout_config.py` and `project/tools/make_extrinsic_noise_config.py`.

6. Live ROS 2 run: see [ros2/live_model/README.md](ros2/live_model/README.md).

The exact scripts used for every run, with their machine-specific paths, are in `project/scripts/`.
