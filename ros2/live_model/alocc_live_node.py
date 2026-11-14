"""Live ALOcc ROS 2 node (system Python 3.10, ROS 2 Humble).

Subscribes to the 5 Waymo scene 0 camera topics from Simi's replay node,
sends each complete frame to the ALOcc model server (Python 3.8, GPU) over a
local socket, and publishes the predicted OccupancyGrid3D on
/alocc/live/occupancy. Records per-frame latency and CPU, GPU and memory use.
"""
import csv, json, os, pickle, socket, struct, subprocess, threading, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from alocc_interfaces.msg import OccupancyGrid3D
from alocc_ros.prediction_types import OccupancyPrediction
from alocc_ros.message_conversion import occupancy_to_msg
from alocc_ros.waymo_scene0_dataset import (CAMERA_SPECS, EGO_FRAME_ID, image_topic,
    load_replay_scene, occupancy_labels, ros_stamp)
from alocc_ros.waymo_scene0_replay_node import RELIABLE_QOS

SOCK = "/tmp/alocc_server.sock"
OUT_DIR = os.path.expanduser("~/work/ros_live/results")
TICK = os.sysconf("SC_CLK_TCK")


def cpu_ticks(pid):
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return int(f[11]) + int(f[12])


def rss_mb(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return 0.0


class Sampler(threading.Thread):
    """Samples GPU use and CPU/RAM use of both processes every 0.5 s."""

    def __init__(self, pids):
        super().__init__(daemon=True)
        self.pids, self.rows, self.stop = pids, [], False

    def run(self):
        # One long-running nvidia-smi streaming every 500 ms (spawning it
        # repeatedly on WSL stalls the GPU driver).
        smi = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits", "-lms", "500"],
                               stdout=subprocess.PIPE, text=True)
        last = {p: cpu_ticks(p) for p in self.pids.values()}
        lt = time.monotonic()
        for line in smi.stdout:
            if self.stop:
                break
            g = line.split(",")
            now = time.monotonic()
            row = {"gpu_util_pct": float(g[0]), "gpu_mem_used_mb": float(g[1])}
            for name, p in self.pids.items():
                t = cpu_ticks(p)
                row[f"{name}_cpu_pct"] = (t - last[p]) / TICK / max(now - lt, 1e-3) * 100
                row[f"{name}_rss_mb"] = rss_mb(p)
                last[p] = t
            lt = now
            self.rows.append(row)
        smi.terminate()


class AloccLiveNode(Node):
    def __init__(self):
        super().__init__("alocc_live_node")
        ds = os.path.expanduser(self.declare_parameter("dataset_path", "~/alocc-data/waymo_scene0").value)
        cfg = self.declare_parameter("model_config", "").value
        self.scene = load_replay_scene(ds, cfg)
        self.ref = self.scene.occupancy
        self.frame_by_stamp = {ros_stamp(f.timestamp_us): f for f in self.scene.frames}
        self.buf, self.first_rx, self.rows = {}, {}, []
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.connect(SOCK)
        self.pub = self.create_publisher(OccupancyGrid3D, "/alocc/live/occupancy", RELIABLE_QOS)
        for cam, suffix in CAMERA_SPECS:
            self.create_subscription(Image, image_topic(suffix),
                                     lambda m, c=cam: self.on_image(c, m), RELIABLE_QOS)
        self.sampler = None
        self.get_logger().info("live node ready, waiting for camera images")

    def rpc(self, obj):
        d = pickle.dumps(obj, protocol=4)
        self.sock.sendall(struct.pack("!Q", len(d)) + d)

        def n(k):
            b = bytearray()
            while len(b) < k:
                b += self.sock.recv(k - len(b))
            return bytes(b)
        return pickle.loads(n(struct.unpack("!Q", n(8))[0]))

    def on_image(self, cam, msg):
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        self.first_rx.setdefault(key, time.perf_counter())
        self.buf.setdefault(key, {})[cam] = (msg.height, msg.width, bytes(msg.data))
        if len(self.buf[key]) == len(CAMERA_SPECS):
            self.run_frame(key, self.buf.pop(key))

    def run_frame(self, key, images):
        frame = self.frame_by_stamp[key]
        t_ready = time.perf_counter()
        r = self.rpc({"order": frame.order, "images": images})
        t_pred = time.perf_counter()
        if self.sampler is None:
            self.sampler = Sampler({"ros_node": os.getpid(), "model_server": r["pid"]})
            self.sampler.start()
        pred = OccupancyPrediction(
            spatial=self.scene.spatial, class_names=self.scene.class_names,
            semantic_labels=occupancy_labels(r["pred"]), timestamp_sec=key[0], timestamp_nanosec=key[1],
            is_mock=False, source="python", timestamp_available=True, sample_index=frame.order,
            sample_metadata=f"dataset=waymo;scene_idx=0;order={frame.order};live_alocc=true",
            flow_supplied=False)
        self.pub.publish(occupancy_to_msg(pred, EGO_FRAME_ID))
        t_pub = time.perf_counter()
        agree = float((r["pred"] == self.ref[frame.order]).mean() * 100)
        row = {"order": frame.order, "frame_idx": frame.frame_idx,
               "image_rx_ms": (t_ready - self.first_rx.pop(key)) * 1e3,
               "model_prep_ms": r["prep_ms"], "model_infer_ms": r["infer_ms"],
               "server_roundtrip_ms": (t_pred - t_ready) * 1e3,
               "convert_publish_ms": (t_pub - t_pred) * 1e3,
               "end_to_end_ms": (t_pub - t_ready) * 1e3,
               "gpu_peak_alloc_mb": r["gpu_peak_mb"], "match_saved_pct": agree}
        self.rows.append(row)
        self.get_logger().info(f"frame {frame.order:2d}: end-to-end {row['end_to_end_ms']:.0f} ms "
                               f"(model {r['infer_ms']:.0f} ms), matches saved {agree:.2f}%")
        if len(self.rows) == len(self.scene.frames):
            self.finish()

    def finish(self):
        self.sampler.stop = True
        self.sampler.join()
        os.makedirs(OUT_DIR, exist_ok=True)
        for name, rows in (("per_frame.csv", self.rows), ("resource_samples.csv", self.sampler.rows)):
            with open(f"{OUT_DIR}/{name}", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

        def stat(rows, k, skip=0):
            v = np.array([x[k] for x in rows[skip:]])
            return {"mean": round(float(v.mean()), 1), "median": round(float(np.median(v)), 1),
                    "p95": round(float(np.percentile(v, 95)), 1), "max": round(float(v.max()), 1)}
        S = self.sampler.rows
        e2e = stat(self.rows, "end_to_end_ms", 1)
        summary = {
            "frames": len(self.rows),
            "note": "first frame excluded from latency stats (CUDA warm-up)",
            "latency_ms": {k: stat(self.rows, k, 1) for k in (
                "end_to_end_ms", "model_infer_ms", "model_prep_ms", "server_roundtrip_ms",
                "convert_publish_ms", "image_rx_ms")},
            "throughput_fps": round(1000 / e2e["mean"], 2),
            "resources": {k: stat(S, k) for k in S[0]},
            "gpu_peak_alloc_mb_torch": round(max(x["gpu_peak_alloc_mb"] for x in self.rows)),
            "frames_matching_saved_100pct": sum(x["match_saved_pct"] == 100.0 for x in self.rows),
        }
        with open(f"{OUT_DIR}/summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        self.get_logger().info("DONE end-to-end ms: " + json.dumps(e2e))
        self.rpc({"cmd": "quit"})
        raise SystemExit


def main():
    rclpy.init()
    node = AloccLiveNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
