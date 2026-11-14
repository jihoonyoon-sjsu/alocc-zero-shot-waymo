# Prediction interface

The framework keeps one internal representation. Connectors and ROS messages both convert into it.

## Occupancy

`OccupancyPrediction` carries:

- voxel dimensions `dim_x`, `dim_y`, `dim_z`
- semantic class names and one integer label per voxel
- spatial resolution and origin
- coordinate-convention string
- timestamp
- `is_mock` and `source`

Labels are row-major, with x changing fastest, then y, then z (`label_layout = x_fastest_then_y_then_z`). A label must be an index into `class_names`.

## Flow

`FlowPrediction` carries the same spatial metadata plus three finite values per voxel, ordered `vx, vy, vz` (`vector_layout = xyz_per_voxel_x_fastest`). `flow_units` is a string from configuration.

The default flow grid matches the occupancy grid. `config/model.yaml` can override flow dimensions, resolution, origin, or coordinate convention when Person 2 confirms that the two outputs differ.

## Validation

`validate_prediction_pair()` checks both objects and then compares their timestamps.

Occupancy:

- `dim_x`, `dim_y`, and `dim_z` are positive integers no larger than 64 for mock, JSON, Python, and ROS data. A saved ALOcc pickle may use up to 256 on each axis because the saved occupancy grids are 200 x 200 x 16
- when geometry is confirmed, resolution values are positive and finite and the origin is finite. A pickle whose shape is not the confirmed grid leaves those fields unset
- `semantic_labels` has shape `(dim_x, dim_y, dim_z)`, stored x-fastest, then y, then z
- every label is an integer index into `class_names`
- class names are unique, non-empty, and at most 256 entries

Flow:

- the same dimension, resolution, and origin checks on the flow grid
- `flow_vectors` has shape `(dim_x, dim_y, dim_z, 3)`, stored as `vx, vy, vz` with x fastest
- every flow value is finite
- `flow_units` is a non-empty string

A pair is rejected when the occupancy and flow timestamps differ by more than the allowed tolerance. File and Python mode use a tolerance of 0, so both outputs in one sample share one stamp. ROS mode uses `ros_input.sync_tolerance_ns`. The two grids may differ in size. Each grid is checked against its own shape.

The ROS validator node runs the same structural checks after transport.

## Option A: prediction files

Input is `file.path` in the model config.

| Setting | Meaning |
| --- | --- |
| `file.path` | One `.json` file, or a directory of prediction files |
| `file.pattern` | Directory glob, default `*.json`. Ignored when `path` is a file |
| `file.loop` | Default true. Read the file or directory again after the last sample. False stops with `ConnectorUnavailable` |

Directory files are loaded in lexicographic filename order, one file per call. Zero-pad names such as `0001.json` when numeric order matters. Paths are resolved from the process working directory.

Each file is one JSON object. Unknown fields, including `sample_id`, are ignored. Both outputs use the same `spatial` object and the same timestamp.

```json
{
  "is_mock": false,
  "timestamp_sec": 3,
  "timestamp_nanosec": 4,
  "spatial": {
    "dim_x": 2,
    "dim_y": 2,
    "dim_z": 1,
    "resolution_x": 0.5,
    "resolution_y": 0.5,
    "resolution_z": 0.5,
    "origin_x": 0.0,
    "origin_y": 0.0,
    "origin_z": 0.0,
    "coordinate_convention": "UNCONFIRMED_PLACEHOLDER_XYZ"
  },
  "class_names": ["unlabeled", "free", "occupied"],
  "semantic_labels": [0, 1, 2, 1],
  "flow_vectors": [0.01, -0.02, 0.005, 0.02, -0.04, 0.005, 0.01, -0.02, 0.005, 0.02, -0.04, 0.005],
  "flow_units": "UNCONFIRMED_PLACEHOLDER_METERS_PER_SECOND",
  "label_layout": "x_fastest_then_y_then_z",
  "vector_layout": "xyz_per_voxel_x_fastest"
}
```

Output of every successful `load()` is `(OccupancyPrediction, FlowPrediction)` with `source="file"`. `is_mock` follows the file and defaults to false. The `spatial` block in the file is the published grid. File mode does not resize that grid to the occupancy dimensions in `model.yaml`. Those YAML dimensions still have to be valid because every mode checks the config, and the mock generator uses them.

A file that fails dimension, label, flow-shape, or timestamp validation raises `ConnectorUnavailable` and is not published. The connector then moves to the next file.

This schema is the file contract. It is not an ALOcc, nuScenes, or Waymo dump.

```bash
./scripts/run_demo.sh --mode file --model-config /path/to/model.yaml
```

## Saved ALOcc pickle playback

`--mode pkl` reads one trusted local `.pkl` from `pkl.path`. The JSON file connector is unchanged.

nuScenes files are a list of dictionaries. The reader uses `pred_occupancy`, `index`, and `pred_flow`. Waymo files are a list of dictionaries, or one dictionary, with `scene_idx`, `frame_idx`, and `pred_occupancy`.

`pred_occupancy` must be a three-axis `uint8` array. `dim_x`, `dim_y`, and `dim_z` are pickle axes 0, 1, and 2. Labels are stored with axis 0 fastest, so `array[i, j, k]` is preserved. The array is not transposed.

When that shape is `confirmed_alocc` `(200, 200, 16)`, the sample uses the 18 nuScenes class names in that block. Waymo predictions use the same ids. Class 0 is `others` and class 17 is `free`. Resolution is 0.4 m and the origin is the minimum corner `(-40, -40, -1)` in the ego frame. The ROS frame id is `ego`. Axis order is `[x, y, z]`: x forward, y left, z up. No extra rotation is applied. A voxel center is `origin + (index + 0.5) * 0.4`.

```text
x = -40.0 + (i + 0.5) * 0.4
y = -40.0 + (j + 0.5) * 0.4
z = -1.0 + (k + 0.5) * 0.4
```

A different shape keeps provisional names `unconfirmed_label_<id>` and does not receive this meter geometry. A confirmed-shape sample whose labels fall outside 0..17 is skipped.

The header stamp is 0 and `timestamp_available` is false. `sample_metadata` includes `timestamp=unavailable`. That stamp is not a capture time, and it is not the ROS publication time. `sample_index` is the position in the file. nuScenes `index`, and Waymo `scene_idx` / `frame_idx`, are copied when present. This repository does not match those Waymo ids to the supporting scene archive, because no verified frame-to-prediction mapping is stored with the pickle.

The motion-flow head is outside the current project scope. Missing, null, or non-3-component flow is not published and is not replaced with zeros. Mock flow is a separate demonstration and is labelled `is_mock`. `pkl.loop` defaults to false. `pkl.max_samples` of 0 reads every usable sample.

```bash
./scripts/run_demo.sh --mode pkl --input /path/to/baseline.pkl --max-samples 8
```

## Option B: Python function

Input is `python.callable`, set to `module:function`.

| Setting | Meaning |
| --- | --- |
| `python.callable` | Import path `module:function` |
| `python.sample` | Optional mapping merged into every sample |
| `python.sample_path` | Optional JSON file, or a directory of JSON sample files |
| `python.pattern` | Directory glob, default `*.json` |
| `python.loop` | Default true. Repeat the sample list after the last sample |

A sample file is one JSON object:

```json
{
  "sample_id": "synthetic-scene-7-frame-0",
  "timestamp_sec": 1700000000,
  "timestamp_nanosec": 0,
  "metadata": {"scene_token": "synthetic-scene-7"},
  "payload": {"split": "test"}
}
```

Fields other than `sample_id`, `timestamp_sec`, `timestamp_nanosec`, `metadata`, and `payload` are copied into `payload`. File values override `python.sample`. `metadata` and `payload` are merged key by key. The connector does not interpret `payload` as occupancy or flow.

A function with one parameter keeps the original call:

```python
def predict(model_config: dict) -> tuple[OccupancyPrediction, FlowPrediction]:
    ...
```

A function that accepts a second positional parameter receives a `PredictionSample`:

```python
def predict(model_config: dict, sample: PredictionSample) -> tuple[OccupancyPrediction, FlowPrediction]:
    ...
```

Output must be `(OccupancyPrediction, FlowPrediction)` with `source="python"` on both objects. Timestamps must be equal. The objects' own dimensions, semantic labels, and flow vector shapes are validated before publishing. `model_config` is passed through so the function can read the configured grid, but the connector does not reshape the returned arrays to that grid. The connector does not import a module until this mode is configured, and it does not call an ALOcc implementation of its own.

## Option C: ROS 2 input topics

Input messages are `alocc_interfaces/OccupancyGrid3D` on `ros_input.occupancy_topic` and `alocc_interfaces/OccupancyFlow3D` on `ros_input.flow_topic`. The defaults are `/alocc/input/occupancy` and `/alocc/input/flow`. Those topics are separate from the framework outputs `/alocc/occupancy` and `/alocc/flow`.

| Setting | Meaning |
| --- | --- |
| `ros_input.sync_tolerance_ns` | Maximum absolute header-stamp difference for a pair. Default 0 |
| `ros_input.wait_timeout_sec` | Drop an unpaired message this long after it arrives. Default 1.0. Zero disables age expiry |
| `ros_input.max_queue` | Maximum unmatched messages retained per topic. Default 16. The oldest arrival is dropped |

Pairing rules:

- The node executor must be spinning. `load()` never calls `spin_once`.
- A later message with the same topic and stamp replaces the earlier message.
- A pair is the oldest occupancy stamp that has a flow message with the same `frame_id` and a stamp within the tolerance. The closest flow stamp wins.
- Occupancy without flow, or flow without occupancy, stays buffered. `load()` raises `ConnectorUnavailable` and names the missing side.
- Stamps outside the tolerance stay buffered until a match arrives, the age timeout drops them, or the queue limit drops the oldest arrival.
- Each message is validated on its own grid. A bad dimension, label list, or flow shape is discarded and reported, including the expected `(dim_x, dim_y, dim_z)` or `(dim_x, dim_y, dim_z, 3)` shape.
- The published pair uses `source="ros"`. Header stamps are copied from the input messages.

Message fields match the internal objects: header stamp and frame id, `is_mock`, `prediction_source`, dimensions, resolution, origin, coordinate convention, then either class names plus `uint8` labels or `float32` flow values. `prediction_source` on the input may be any allowed source. The connector rewrites the published source to `ros`.

```bash
./scripts/run_demo.sh --mode ros --model-config /path/to/model.yaml
```

## Mock generator

`generate_mock_pair()` fills labels with `(x + 2y + 3z) mod class_count` and flow with `(0.01(x+1), -0.02(y+1), 0.005(z+1))`. The pattern does not depend on the clock. The timestamp defaults to the current time so latency measurement has a real stamp. Every mock object has `is_mock=True` and `source="mock"`.
