# Live ALOcc-3D inside ROS 2 (Waymo scene 0)

This plugs the real ALOcc-3D model into the `alocc_ros` replay. The replay
node publishes the 5 camera images. `alocc_live_node.py` collects each frame
and sends it to `alocc_server.py`, which runs ALOcc on the GPU and returns the
200x200x16 grid. The node publishes it on `/alocc/live/occupancy`
(`OccupancyGrid3D`, source "python").

Why two programs: ROS 2 Humble needs Python 3.10 and the ALOcc env needs 3.8.
They talk over a local Unix socket (`/tmp/alocc_server.sock`).

Only the images come through ROS. Calibration, poses and the 16-frame history
come from the Waymo adapter (`project/tools/waymo_zero_shot.py`, scene 0).
The server is hard-wired to scene 0.

## Files
- `alocc_server.py`: model server (alocc conda env)
- `alocc_live_node.py`: ROS 2 node, plus latency, GPU, CPU and RAM sampling
- `run_live.sh`: starts the server, the live node, and the replay node
- `test_server.py`: check without ROS (sends the zip images, compares with the saved predictions)
- `results/`: the Nov 13 run (`summary.json`, `per_frame.csv`, `resource_samples.csv`, `live_run.log`)

## Result (RTX 4070 Ti, WSL Ubuntu 22.04, ROS 2 Humble; frame 0 left out as warm-up)
- 40/40 frames identical to the saved zero-shot predictions.
- End-to-end per frame (images in to grid published): mean 812 ms, p95 915 ms (about 1.2 FPS).
  - Model compute is 431 ms, image prep 185 ms, and grid to ROS message 122 ms.
- Peak GPU memory 8.4 GB. RAM: about 2.7 GB for the server, 0.2 GB for the node.
- The replay ran at `replay_rate 0.2` (2.5 s per frame) so no frames drop. At full speed (0.5 s per frame) the model cannot keep up.
- The ~1.4 s the replay node takes to publish all 5 images is not counted in the end-to-end time.

This measures latency and resources, not accuracy. Accuracy comes from the offline runs.
