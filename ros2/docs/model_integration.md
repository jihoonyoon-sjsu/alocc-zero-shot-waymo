# Connecting Person 1 and Person 2

Person 1 owns the modified ALOcc model and robustness experiments. Person 2 runs nuScenes and Waymo through that model and produces predictions. This repository is the integration point. It does not implement either of those pieces.

## Three ways to deliver predictions

### A. Prediction files

Person 2 writes one JSON object per sample. `file.path` can be that file or a directory of `*.json` files. A directory is read in filename order, one file per `load()`. `file.loop` defaults to true, so a single file is read again on the next tick and a directory restarts after the last file. Set `file.loop` to false to stop when the sequence ends.

The JSON contract is in `docs/prediction_interface.md`. `--mode file` refuses to start when `file.path` is empty, missing, or a directory with no matching files. This repository does not parse an ALOcc checkpoint or the nuScenes and Waymo datasets themselves. Person 2's exporter has to write this JSON, or the file connector is the only module that should change. Saved prediction pickles are a separate playback path, `--mode pkl`, documented in `docs/prediction_interface.md`.

### B. Python inference function

Person 2 provides an importable function:

```text
python.callable: their_package.inference:predict
```

`predict(model_config)` remains valid. If the function accepts a second parameter, the connector calls `predict(model_config, sample)` with a `PredictionSample`. The sample comes from `python.sample`, from `python.sample_path` (one JSON file or a directory of JSON files), or from both, with the file values overriding the YAML values. No model code is imported until `python.callable` is set.

Returned objects must be `OccupancyPrediction` and `FlowPrediction` with `source="python"`. Their timestamps must match. Dimensions, semantic labels, and flow vectors are validated before anything is published.

### C. ROS 2 topics

Person 2 publishes `alocc_interfaces/OccupancyGrid3D` and `alocc_interfaces/OccupancyFlow3D` on `ros_input.occupancy_topic` and `ros_input.flow_topic`. The connector pairs a message only with the other side when the frame ids are equal and the header stamps differ by at most `ros_input.sync_tolerance_ns` (default 0). If one side is missing, that message stays buffered. An unpaired message older than `ros_input.wait_timeout_sec` is dropped. `ros_input.max_queue` bounds how many unmatched messages are kept on each topic. `load()` does not spin; the node executor delivers the callbacks.

A message whose dimensions, labels, or flow shape fail validation is discarded and reported. Downstream publishers can keep using the internal format. If Person 2 chooses different message types, the ROS connector is the only module that should change.

Do not publish Person 2's raw stream directly on `/alocc/occupancy` or `/alocc/flow` unless it already matches those messages. The demo subscribers, validator, monitor, and visualization node consume those output topics.

## Confirmed occupancy contract

Person 2 confirmed the occupancy grid used by both nuScenes and Waymo predictions. It is stored in `config/model.yaml` under `confirmed_alocc` and applied by pickle playback when the array shape matches.

- 18 class ids, in order: `others`, `barrier`, `bicycle`, `bus`, `car`, `construction_vehicle`, `motorcycle`, `pedestrian`, `traffic_cone`, `trailer`, `truck`, `driveable_surface`, `other_flat`, `sidewalk`, `terrain`, `manmade`, `vegetation`, `free`
- Shape `(200, 200, 16)` in `[x, y, z]` order
- Voxel size 0.4 m, origin `(-40, -40, -1)`, bounds x and y `[-40, 40]`, z `[-1, 5.4]`
- Ego frame: x forward, y left, z up
- Class 17, `free`, is empty space

The motion-flow head is outside the current project scope. This repository does not generate flow, does not turn a two-component array into a three-component ROS flow message, and does not publish flow for an occupancy-only pickle. The mock generator still publishes flow so the original demonstration keeps working. Those mock vectors are not ALOcc motion flow.

Sensor timestamps and a Waymo frame-to-archive mapping are still not available from the prediction files.

## Information still needed from Person 1

- Whether a future flow head is scene flow or object motion, and its units
- Timestamp convention, if a later export includes capture times

## Information needed from Person 2

- Which of the three delivery methods they will use
- For files: path layout, file extension, one sample versus a sequence, and how the timestamp is stored
- For Python: module path, constructor or weight-loading step, and the conda or venv they expect
- For ROS: topic names, message types, QoS, frame id, and publish rate
- Sample identifiers they want preserved, such as scene and frame tokens
- Confirmation that their arrays use the layout strings `x_fastest_then_y_then_z` and `xyz_per_voxel_x_fastest`, or the actual layout if it differs
- A single small real sample that can replace the mock generator in a regression test

`config/model.yaml` `occupancy` keeps the small mock class list `unlabeled`, `free`, and `occupied`. Confirmed ALOcc class names live in `confirmed_alocc`. Flow units stay `UNCONFIRMED_PLACEHOLDER_METERS_PER_SECOND` because flow is outside this milestone.

## What this repository will not do

- Install the ALOcc training environment
- Download nuScenes or Waymo
- Run Person 2's preprocessing or official evaluation
- Report mock FPS or latency as model inference performance
