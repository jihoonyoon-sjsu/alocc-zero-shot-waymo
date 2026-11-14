# Pipeline validation checks

Before reading the Waymo drop as a real transfer failure, we checked that our setup was feeding the model correct inputs and scoring it correctly. All checks passed. Scripts are in `project/scripts/checks/`.

Some checks printed their result to the terminal and were not saved as files. For those, the result below is the one recorded at the time.

| # | Check | Script | Result | Evidence |
|---|---|---|---|---|
| 1 | **Index alignment.** Occ3D-Waymo scene index = position in the sorted Waymo v2 segment names, and frame index = sorted timestamp order. | `check_waymo_alignment.py` | Scenes 0, 1, 100 and 201: ego poses identical (0.000 m difference) and frame counts equal. | Terminal output (not saved). Check 7 later confirmed the same alignment on all 202 scenes. |
| 2 | **Camera slot order.** Which Waymo camera each of the 5 slots in Occ3D's pose file is. | `check_waymo_cam_order.py`, `slot_report.py` | Slots 0 to 4 = FRONT, FRONT_LEFT, SIDE_LEFT, FRONT_RIGHT, SIDE_RIGHT, matched by extrinsics and focal length. Yaws -0.1, 45.3, 90.2, -45.1, -89.9 degrees. ALOcc's own code labels slots 2 and 3 the other way round; our adapter fixes the labels. | `camera_slots.txt` |
| 3 | **Adapter output.** What the model actually receives from `waymo_zero_shot.py`. | `check_waymo_dataset.py` | 2 Hz subsampling gives 8,069 frames. Ego motion between kept frames is about 9.08 m per 0.5 s, and 0 at each scene start. Intrinsics converted to standard K (f about 2050 px). Camera optical axes point at yaws 0, ±45, ±90 degrees. | Terminal output (not saved). Checks 4 and 8 confirm the end result. |
| 4 | **Smoke test** on scene 0 (40 frames), including a grid orientation test. | `check_waymo_smoke.py` | Occupied IoU 48.5. Share of GT vehicle voxels predicted as vehicle: 36.4% as-is vs 8.3% (flip x), 1.7% (flip y), 1.7% (transpose), so the grid orientation is right. | `smoke_check_scene0.txt` |
| 5 | **Chunk merge** does not change results. Long runs were split at scene boundaries and merged. | `check_baseline_merge.py` | The merged nuScenes baseline scores 45.46 mIoU, with every class equal to the unchunked run. | `../nuscenes/baseline.padcheck.log` |
| 6 | **History reset** really changes predictions. | `check_reset_differs.py` | Chunk 0: 2,034 frames, only the 51 first-of-scene frames identical (as expected, they have no history in either run). Mean 35.33% of voxels differ on the rest. | `history_reset_vs_zero_shot_chunk0.txt` |
| 7 | **Image and calibration pairing**, all 202 scenes. | `check_pairing.py` | 0 timestamp mismatches, 0 calibration mismatches. | `image_and_calibration_pairing_all_scenes.txt` |
| 8 | **Ground truth projected onto the model's input images** (after resize and crop), with Occ3D's calibration and separately with Waymo v2's own calibration. | `project_gt_on_images.py`, `dump_v2calib.py`, `project_gt_v2_calib.py` | GT voxels land on the right objects in all 5 cameras with both calibrations. | Figures kept off GitHub because they contain Waymo camera images (Waymo dataset license). Available from the team on request. |

## Scorer checks

- **nuScenes scorer** (`score_nuscenes_merged.py`) reproduces ALOcc's own occupied IoU (75.30) from the baseline prediction file.
- **Waymo scorer** (`score_waymo.py`) asserts the prediction file has exactly the 8,069 expected (scene, frame) pairs, each once, with shape 200×200×16 and valid labels. It reproduces the smoke test on scene 0 (48.52).

## Independent review

An outside code review (Codex) of the camera conversion, resize and crop, 2 Hz selection, ego motion, history reset, mask, class map and chunk deduplication found no errors. Its three requests (verify slot pairing on all scenes, add the frame-coverage check, don't claim camera height as a proven cause) were all done.
