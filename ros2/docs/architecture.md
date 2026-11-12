# Architecture

```text
Mock prediction generator
        |
        v
Prediction input interface
        |
        v
prediction_input_node
        +------------------+
        |                  |
        v                  v
 Occupancy output      Flow output
        |                  |
        v                  v
 /alocc/occupancy      /alocc/flow
        |                  |
        +---------+--------+
                  |
                  v
          visualization_node
          SVG slices and RViz markers
                  |
                  v
           runtime_monitor
                  |
                  v
         results/runtime/*.csv
```

`occupancy_subscriber`, `flow_subscriber`, and `prediction_validator` also subscribe to the two output topics. They record receipt and structural validation.

## Packages

`alocc_interfaces` defines the wire format:

- `OccupancyGrid3D`
- `OccupancyFlow3D`

`alocc_ros` is the Python integration package. Nodes stay behind the connector interface, so a future file reader, Python function, or ROS subscription can return the same internal objects.

## Nodes

| Executable | Role |
| --- | --- |
| `prediction_input_node` | Loads one pair, validates it, publishes both topics, and records stage timings |
| `occupancy_publisher` | Publishes occupancy only. Not started by the combined demo |
| `flow_publisher` | Publishes flow only. Not started by the combined demo |
| `occupancy_subscriber` | Records received occupancy messages |
| `flow_subscriber` | Records received flow messages |
| `prediction_validator` | Converts messages back to internal objects and validates them |
| `runtime_monitor` | Computes latency, FPS, and memory, then writes `summary.csv` |
| `visualization_node` | Writes SVG slices and publishes `visualization_msgs/MarkerArray` |

The demo launch file starts the input node plus the subscribers, validator, monitor, and visualization node. It does not start the standalone publishers, because that would publish a second pair on the same topics.

## Replaceable input

`PredictionConnector.load()` returns `(OccupancyPrediction, FlowPrediction | None)`. `None` means the source did not supply usable flow. The pickle connector is the path that can return `None`. Mock, JSON, Python, and ROS input still return both predictions.

| Connector | Module | Behavior today |
| --- | --- | --- |
| Mock | `connectors/mock.py` | Deterministic synthetic grids, always `is_mock=True` |
| File | `connectors/file_input.py` | Reads one documented JSON file, or each JSON file in a directory in filename order |
| Python | `connectors/python_inference.py` | Calls `predict(model_config)` or `predict(model_config, sample)` when configured |
| ROS | `connectors/ros_subscription.py` | Pairs input messages by frame id and timestamp, and keeps a missing side buffered |

Unavailable connectors raise `ConnectorUnavailable` with an explicit reason. They do not invent dataset samples or model outputs.

## Runtime data

`prediction_input_node` writes `results/runtime/stage_timings.csv` with prediction loading time, conversion time, and local publish-call time.

`runtime_monitor` writes `results/runtime/latency.csv` and `results/runtime/summary.csv`. End-to-end latency is receive time minus the message stamp. Every summary row uses the measurement context `mock_pipeline_not_model_inference`.

## Visualization

The SVG renderer has no GUI dependency. RViz displays `/alocc/visualization/markers` with fixed frame `ego` when a display is available. Real occupancy uses one `CUBE_LIST` marker per semantic class in the `occupancy` namespace. Free space, class `free`, is not drawn. Class `others` is drawn. Mock cubes stay in `mock_occupancy`, and mock flow arrows stay in `mock_flow`.

Colors are stable: car blue, truck dark blue, bus yellow, pedestrian red, bicycle cyan, motorcycle magenta, barrier orange, traffic cone red-orange. The full list is logged once by `visualization_node` and defined in `occupancy_geometry.py`.

`visualization.downsample_stride` of 1 and `visualization.max_voxels` of 0 keep every non-free voxel. A larger stride keeps indexes divisible by that stride on x, y, and z. A positive cap walks x-fastest order and leaves out later voxels. The visualization subscription keeps one reliable message, and the cube list is built on a worker thread so marker construction does not stall the occupancy publisher. Large occupancy arrays are not sent best-effort, because that transport dropped them.
