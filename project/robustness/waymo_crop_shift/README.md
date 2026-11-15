# Waymo crop shift experiment (Tara's robustness code)

Zero-shot ALOcc-3D (trained on nuScenes, no fine-tuning) tested on the full Waymo validation set: 8,069 frames (every 5th frame of 202 scenes). The only change from the baseline is the test crop position, applied with `RobustPrepareImageInputs(crop_shift=N)` from `project/robustness/robustness_pipeline.py`.

## Question

The Waymo horizon sits at about row 22 of the 256-row model input. On nuScenes it sits at about row 62. If that mismatch hurts transfer, moving the crop window up should help:
- 40 px up puts the Waymo horizon at about row 62, the same as nuScenes.
- 50 px up puts it at about row 72.

## Results (8 merged classes, same scorer as the baseline)

| | Original crop (baseline) | Shift 40 | Shift 50 |
|---|---|---|---|
| mIoU | **9.56** | 8.81 | 8.70 |
| Occupied IoU | **22.48** | 21.43 | 21.22 |
| vehicle | **6.55** | 5.61 | 5.46 |
| pedestrian | **2.10** | 1.52 | 1.46 |
| two_wheeler | **1.28** | 0.93 | 1.07 |
| traffic_cone | **1.03** | 0.81 | 0.77 |
| drivable_surface | **35.40** | 33.48 | 33.09 |
| sidewalk | **15.06** | 13.71 | 13.63 |
| vegetation | **8.12** | 7.92 | 7.78 |
| manmade | **6.94** | 6.47 | 6.37 |
| mIoU 0-10 m | **10.08** | 8.10 | 7.89 |
| mIoU 10-20 m | **13.14** | 11.46 | 11.28 |
| mIoU 20-30 m | **10.00** | 9.27 | 9.12 |
| mIoU 30-40 m | **8.34** | 8.00 | 7.93 |
| mIoU 40 m+ | **6.17** | 5.82 | 5.82 |

Sector mIoU (shift 40 / shift 50): FRONT 10.08 / 9.93, FRONT_LEFT 7.95 / 7.92, SIDE_LEFT 7.69 / 7.58, FRONT_RIGHT 8.81 / 8.68, SIDE_RIGHT 8.13 / 8.00. The baseline sectors are in `scores/baseline_original_crop.score.json`.

## What it means

- Both shifts score below the original crop in every class except two_wheeler at shift 50, and in every distance band. Even shift 40, which lines the horizon up with nuScenes, does not help.
- So these results do not support horizon position as something a crop shift can fix.
- The biggest loss is close to the car (0-10 m). A likely reason is that shifting the crop up cuts off the bottom of the image, where the nearby road is.
- **Limit:** a crop shift changes what part of the scene is visible as well as where the horizon sits. This does not rule out camera geometry as a cause in general.

## Checks

- **Shift 0 changes nothing:** `crop_shift_0` on Waymo scene 0 gave 57/57 prediction entries identical to the saved baseline predictions. So any difference above comes from the shift alone.
- **Full data:** every run used all 8,069 frames, the same as the baseline. All chunks exited 0 (`run_logs/progress.txt`).

## Files

- `scores/`: score JSON and text for each run, plus the baseline.
- `configs/`: the run configs. Each one is the zero-shot Waymo config with only the image step swapped.
- `run_tara.sh`: the run script. It runs 4 scene chunks per config, merges them, then scores.
- `run_logs/`: chunk progress and merge logs. The early lines in `progress.txt` are from a 4-run batch (with noise and blur) that was stopped and replaced by these two crop runs.
- The predictions (about 20 GB) are not included. They are on JiHoon's machine at `~/work/results/tara_robustness/preds/`.

**Run setup:** RTX 4070 Ti, WSL Ubuntu 22.04, about 3.2 frames/s, about 37 to 40 min per run. Nov 14 2026.
