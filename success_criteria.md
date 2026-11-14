# Success criteria status

Status review of the seven target criteria, written on `feature/alocc-ros2-waymo-replay` on November 13, 2026. Scores below come from files already in the repository. Percentages are rounded to two decimals from those files.

Four criteria are met. Two are partial. The conference report is not in the repository.

| Criterion | Status |
| --- | --- |
| Reproduce published nuScenes results for ALOcc within run-to-run noise | Met |
| Waymo pipeline passes all validation checks | Partial |
| Zero-shot geometric IoU and merged-class mIoU on Occ3D-Waymo | Met |
| Camera dropout and perturbation ablations for ALOcc | Met |
| Waymo degradation broken down by camera coverage, history, distance, and class | Met |
| A working ROS 2 node with runtime measurements on two platforms | Partial |
| A conference-style report, including negative or mixed results | Not met |

## Reproduce published nuScenes results

Met. The published ALOcc-3D figure recorded in the repository README is 45.5 mIoU. The official evaluation log reports 45.46 mIoU on 6,019 Occ3D-nuScenes validation frames. The same 45.46 appears in `project/results/nuscenes/baseline.merged.log` and in the pad-check log listed by `project/results/diagnostics/all_run_miou.txt`. The gap from 45.5 is 0.04 points. Those logs are records of this reproduction, not a separate multi-seed variance study.

The eight-class score for that same baseline is in `project/results/nuscenes/baseline.merged8.score.json`: occupied IoU 75.30, merged-class mIoU 50.25, 6,019 frames.

## Waymo pipeline validation

Partial. The checks that have a saved result look clean:

- Image and calibration pairing, 202 scenes: 0 timestamp mismatches and 0 slot or calibration mismatches. Source: `project/results/diagnostics/image_and_calibration_pairing_all_scenes.txt`.
- Scene 0 smoke prediction: 40 unique frames, occupancy shape 200×200×16. Source: `project/results/diagnostics/smoke_check_scene0.txt`.
- History reset changes the prediction: 2,034 frames, 51 identical, mean 35.33% of voxels differ. Source: `project/results/diagnostics/history_reset_vs_zero_shot_chunk0.txt`.

Scripts for index alignment, the dataset check, camera order, chunk merge, and ground-truth projection are in `project/scripts/checks/`. Those five do not have a saved result under `project/results/diagnostics/`, so this review cannot mark every check as passed.

## Zero-shot Waymo scores

Met. `project/results/waymo/zero_shot.score.json` scores the zero-shot prediction on 8,069 Occ3D-Waymo frames:

- Occupied (geometric) IoU: 22.48
- Merged eight-class mIoU: 9.56

Class IoU: vehicle 6.55, pedestrian 2.10, two-wheeler 1.28, traffic cone 1.03, drivable surface 35.40, sidewalk 15.06, vegetation 8.12, manmade 6.94.

Against the nuScenes eight-class score (occupied IoU 75.30, mIoU 50.25), this is a large drop. The number itself is the criterion, and it is in the repository.

## Camera dropout and perturbation

Met for the nuScenes ablations that have logs. Official mIoU on 6,019 frames, from `project/results/diagnostics/all_run_miou.txt`:

| Run | mIoU |
| --- | --- |
| Baseline | 45.46 |
| Drop CAM_BACK | 40.24 |
| Drop CAM_BACK_LEFT | 43.38 |
| Drop CAM_BACK_RIGHT | 43.51 |
| Drop CAM_FRONT | 38.60 |
| Drop CAM_FRONT_LEFT | 42.74 |
| Drop CAM_FRONT_RIGHT | 42.78 |
| Extrinsic noise level 1 | 38.00 |
| Extrinsic noise level 2 | 38.95 |
| Extrinsic noise level 3 | 26.35 |
| Extrinsic noise level 4 | 32.95 |

Level 3 is worse than level 4, so the noise series is not strictly monotonic. Score files for the eight-class version of these runs are under `project/results/nuscenes/merged8_scores/`.

Gaussian noise, blur, and the upward crop are code on `feature/waymo-robustness-tf`. This branch does not contain score files for those Waymo image corruptions.

## Waymo degradation breakdown

Met. The zero-shot score file splits the 8,069 frames by camera sector, distance, and class. History is a second full run.

Camera sector, occupied IoU and eight-class mIoU:

| Sector | Occupied IoU | mIoU |
| --- | --- | --- |
| FRONT | 28.73 | 10.91 |
| FRONT_RIGHT | 23.68 | 9.65 |
| SIDE_RIGHT | 23.93 | 8.68 |
| FRONT_LEFT | 19.09 | 8.72 |
| SIDE_LEFT | 18.39 | 8.43 |

Distance, occupied IoU: 0–10 m 25.65, 10–20 m 33.70, 20–30 m 23.97, 30–40 m 18.96, 40 m and beyond 15.25.

History reset, `project/results/waymo/history_reset.score.json`, same 8,069 frames: occupied IoU 23.72 and eight-class mIoU 9.28. Compared with zero-shot (22.48 and 9.56), clearing history raises occupied IoU by about 1.2 points and lowers mIoU. History does not account for the Waymo drop.

## ROS 2 node and runtime measurements

Partial. This branch adds a ROS 2 Jazzy package under `ros2/`. The Waymo scene 0 replay node publishes the five camera images, camera info, and the occupancy grid from `~/alocc-data/waymo_scene0`. The dataset stays outside git. Build and test instructions are in `ros2/docs/installation.md` and `ros2/docs/testing.md`. The documented target machine is Ubuntu 24.04 with ROS 2 Jazzy.

On that Ubuntu virtual machine the test suite passed 77 tests, and a 40-frame scene 0 replay completed. Those logs are on the virtual machine. They are not files in this repository.

`ros2/src/alocc_ros/alocc_ros/runtime_monitor.py` measures latency, rate, and memory for the mock pipeline. The node states that those timings are not ALOcc inference latency or FPS. The docs name one platform. There is no second-platform runtime table in the repository, so the two-platform measurement is still open.

## Conference-style report

Not met. The repository has no paper, TeX, or PDF. The negative and mixed results are already computed: the Waymo occupied IoU drop from 75.30 to 22.48, the history reset that does not recover mIoU, and extrinsic noise that is worse at level 3 than at level 4. They are not written up as a conference report.

## What is still open

1. Save the five Waymo check scripts that do not yet have a result file.
2. Score the Waymo noise, blur, and crop runs, if those ablations stay in the criteria.
3. Record ROS runtime on a second platform, and keep model-inference time separate from mock-pipeline time.
4. Write the conference report from the score files above.
