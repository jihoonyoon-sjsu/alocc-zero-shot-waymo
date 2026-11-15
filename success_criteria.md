# Success criteria status

Rewritten November 14, 2026 from `main` at `bcc2618`. This replaces the November 13 review. Percentages are rounded to two decimals from the score files. No code or scripts were changed for this update.

Five criteria are met. The ROS 2 runtime criterion is partial. The conference report is not in the repository.

| Criterion | Status |
| --- | --- |
| Reproduce published nuScenes results for ALOcc within run-to-run noise | Met |
| Waymo pipeline passes all validation checks | Met |
| Zero-shot geometric IoU and merged-class mIoU on Occ3D-Waymo | Met |
| Camera dropout and perturbation ablations for ALOcc | Met |
| Waymo degradation broken down by camera coverage, history, distance, and class | Met |
| A working ROS 2 node with runtime measurements on two platforms | Partial |
| A conference-style report, including negative or mixed results | Not met |

What changed since the November 13 review: `project/results/diagnostics/VALIDATION.md` records all eight Waymo checks as passed, the Waymo crop-shift runs are scored, and `ros2/live_model/` has a real ALOcc latency run. Gaussian noise and blur on Waymo still have no score file. There is still no conference paper.

## Reproduce published nuScenes results

Met. The published ALOcc-3D figure recorded in the repository README is 45.5 mIoU. The official evaluation log reports 45.46 mIoU on 6,019 Occ3D-nuScenes validation frames. The same 45.46 is in `project/results/nuscenes/baseline.merged.log` and in `project/results/diagnostics/all_run_miou.txt`. The gap from 45.5 is 0.04 points.

`project/results/diagnostics/VALIDATION.md` adds the chunk-merge check: the merged baseline scores 45.46 mIoU, and every class matches the unchunked run (`project/results/nuscenes/baseline.padcheck.log`). These are records of one reproduction, not a multi-seed variance study.

The eight-class score for that baseline is in `project/results/nuscenes/baseline.merged8.score.json`: occupied IoU 75.30, merged-class mIoU 50.25, 6,019 frames. The validation note says the nuScenes scorer reproduces ALOcc's own occupied IoU of 75.30 from the baseline prediction file.

## Waymo pipeline validation

Met. `project/results/diagnostics/VALIDATION.md` says all checks passed. The table there covers the eight checks:

| Check | Result recorded in VALIDATION.md |
| --- | --- |
| Index alignment | Scenes 0, 1, 100, and 201: ego poses identical (0.000 m) and frame counts equal. Terminal output was not saved. Check 7 later confirmed alignment on all 202 scenes. |
| Camera slot order | Slots 0 to 4 are FRONT, FRONT_LEFT, SIDE_LEFT, FRONT_RIGHT, SIDE_RIGHT. Saved in `camera_slots.txt`. |
| Adapter output | 2 Hz subsampling gives 8,069 frames. Ego motion between kept frames is about 9.08 m per 0.5 s, and 0 at each scene start. Terminal output was not saved. |
| Scene 0 smoke | Occupied IoU 48.5. Grid orientation holds: 36.4% of ground-truth vehicle voxels are predicted as vehicle as-is, against 8.3%, 1.7%, and 1.7% for the three flipped orientations. Saved in `smoke_check_scene0.txt`. |
| Chunk merge | Merged nuScenes baseline is 45.46 mIoU, every class equal to the unchunked run. |
| History reset | 2,034 frames, 51 identical, mean 35.33% of voxels differ. Saved in `history_reset_vs_zero_shot_chunk0.txt`. |
| Image and calibration pairing | 202 scenes, 0 timestamp mismatches, 0 calibration mismatches. |
| Ground truth on the model input images | Ground-truth voxels land on the right objects in all 5 cameras with both calibrations. The figures are not on GitHub because they contain Waymo images. |

The Waymo scorer checks that a prediction file has exactly 8,069 scene-frame pairs, shape 200×200×16, and valid labels. It reproduces the scene 0 smoke test at occupied IoU 48.52. An outside code review of the adapter, named in that same validation note, found no errors.

## Zero-shot Waymo scores

Met. `project/results/waymo/zero_shot.score.json` scores 8,069 Occ3D-Waymo frames:

- Occupied (geometric) IoU: 22.48
- Merged eight-class mIoU: 9.56

Class IoU: vehicle 6.55, pedestrian 2.10, two-wheeler 1.28, traffic cone 1.03, drivable surface 35.40, sidewalk 15.06, vegetation 8.12, manmade 6.94.

Against the nuScenes eight-class score (occupied IoU 75.30, mIoU 50.25), this is a large drop. The crop-shift baseline file `project/robustness/waymo_crop_shift/scores/baseline_original_crop.score.json` repeats the same 22.48 and 9.56.

## Camera dropout and perturbation

Met. Official nuScenes mIoU on 6,019 frames, from `project/results/diagnostics/all_run_miou.txt`:

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

Level 3 is worse than level 4, so the noise series is not strictly monotonic. Eight-class scores for these runs are under `project/results/nuscenes/merged8_scores/`.

Waymo crop shift is now scored on all 8,069 frames (`project/robustness/waymo_crop_shift/README.md`, November 14, 2026, RTX 4070 Ti):

| Crop | Occupied IoU | 8-class mIoU |
| --- | --- | --- |
| Original | 22.48 | 9.56 |
| 40 px up | 21.43 | 8.81 |
| 50 px up | 21.22 | 8.70 |

Both shifts score below the original crop in every distance band. Shift 40, which lines the Waymo horizon up with nuScenes, does not help. The crop note says a shift of 0 on scene 0 matched the saved baseline on 57 of 57 entries, and every chunk exited 0. Gaussian noise and blur are implemented in `project/robustness/image_corruption.py`. The run log says that noise-and-blur batch was stopped and replaced by the crop runs, so those two Waymo corruptions have no score file.

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

The crop-shift result is a further negative finding for one camera-geometry idea: moving the crop so the horizon matches nuScenes lowers both occupied IoU and mIoU. The largest mIoU loss is at 0–10 m (10.08 down to 8.10 at 40 px and 7.89 at 50 px).

## ROS 2 node and runtime measurements

Partial. Two ROS setups are in the repository, and a real model timing table exists for one of them.

The replay package under `ros2/` publishes Waymo scene 0 cameras and occupancy. Its documented machine is Ubuntu 24.04 with ROS 2 Jazzy. `runtime_monitor.py` measures the mock pipeline and states that those timings are not ALOcc inference latency or FPS. This repository does not contain a live-model latency table from that Jazzy machine.

The live node is in `ros2/live_model/`. It was run on November 13, 2026 on WSL Ubuntu 22.04 with ROS 2 Humble and an RTX 4070 Ti. `results/summary.json` and `results/live_run.log` report:

- 40 of 40 frames match the saved zero-shot predictions.
- End-to-end, images in to grid published, frame 0 left out as warm-up: mean 811.9 ms, median 809.8 ms, p95 914.6 ms. About 1.23 FPS.
- Model compute mean 430.5 ms. Image prep mean 184.8 ms. Grid-to-message mean 121.5 ms.
- Torch peak GPU allocation 8,371 MB. Model-server RSS mean about 2,703 MB. ROS node RSS mean about 180 MB.
- The replay used `replay_rate` 0.2 so frames were not dropped. The live note says the model cannot keep up at the full 0.5 s frame interval.

That is a working ROS 2 node with model runtime numbers on one platform. The same measurement on the Jazzy machine is not in the repository, so the two-platform requirement is still open.

## Conference-style report

Not met. The repository has no paper, TeX, or PDF. The negative and mixed results are already computed: the Waymo occupied IoU drop from 75.30 to 22.48, the history reset that does not recover mIoU, extrinsic noise that is worse at level 3 than at level 4, and a crop shift that does not repair the Waymo horizon mismatch. `VALIDATION.md` and `project/robustness/waymo_crop_shift/README.md` are lab notes, not a conference report.

## What is still open

1. Score the Waymo Gaussian noise and blur runs, if those ablations stay in scope. The crop runs replaced that batch.
2. Record the live ALOcc latency table on a second platform. The Humble GPU run is done. The Jazzy machine does not have one in git.
3. Write the conference report from the score files above.
