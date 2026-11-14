"""ALOcc-3D model server (runs in the alocc conda env, Python 3.8).

Loads the model once, then answers requests over a local Unix socket.
Request: {"order": i, "images": {cam: (h, w, rgb_bytes)}}
Reply:   {"order": i, "pred": uint8 (200,200,16), "infer_ms", "gpu_mem_mb"}
Images received from ROS replace the disk reads for that frame (Image.open is
patched), so the model really runs on what ROS delivered. Calibration, poses
and the 16-frame history come from the existing Waymo adapter (scene 0).
"""
import os, sys, time, pickle, socket, struct, io
os.chdir(os.path.expanduser("~/work/ALOcc")); sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from PIL import Image
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate, scatter
from mmcv.runner import load_checkpoint
sys.path.insert(0, os.path.expanduser("~/work/project/tools"))
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
import mmdet3d.datasets.pipelines.loading as loading

SOCK = "/tmp/alocc_server.sock"
ALOCC = os.path.expanduser("~/work/ALOcc")
CFG = os.path.expanduser("~/work/project/configs/alocc_3d_waymo_zero_shot.py")
CKPT = ALOCC + "/ckpts/alocc_3d_r50_256x704_bevdet_preatrain_16f.pth"

received = {}   # absolute image path -> PIL image delivered over ROS
stats = {"ros": 0, "disk": 0}
_open = Image.open
def patched_open(fp, *a, **k):
    key = os.path.realpath(fp) if isinstance(fp, str) else None
    if key in received:
        stats["ros"] += 1
        return received[key].copy()
    stats["disk"] += 1
    return _open(fp, *a, **k)
loading.Image.open = patched_open

def recv_msg(c):
    n = struct.unpack("!Q", recv_n(c, 8))[0]
    return pickle.loads(recv_n(c, n))
def recv_n(c, n):
    b = bytearray()
    while len(b) < n:
        chunk = c.recv(min(n - len(b), 1 << 22))
        if not chunk: raise ConnectionError("closed")
        b += chunk
    return bytes(b)
def send_msg(c, obj):
    d = pickle.dumps(obj, protocol=4)
    c.sendall(struct.pack("!Q", len(d)) + d)

def main():
    os.chdir(ALOCC)
    cfg = Config.fromfile(CFG)
    if cfg.get("custom_imports"):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])
    cfg.data.test.scene_range = [0, 1]
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    dataset = build_dataset(cfg.data.test)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, CKPT, map_location="cpu", revise_keys=[(r"^module\.", ""), (r"^teacher\.", "")])
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model, device_ids=[0]).eval()
    infos = dataset.data_infos
    print(f"model ready, scene 0 frames: {len(infos)}", flush=True)

    if os.path.exists(SOCK): os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK); srv.listen(1)
    while True:
        c, _ = srv.accept()
        print("client connected", flush=True)
        try:
            while True:
                req = recv_msg(c)
                if req.get("cmd") == "quit":
                    send_msg(c, {"ok": True}); return
                i = req["order"]
                info = infos[i]
                for cam, (h, w, buf) in req["images"].items():
                    path = os.path.realpath(info["cams"][cam]["data_path"])
                    received[path] = Image.frombytes("RGB", (w, h), buf)
                t0 = time.perf_counter()
                data = collate([dataset[i]], samples_per_gpu=1)
                data = scatter(data, [0])[0]
                t1 = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    out = model.module(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                r = out[0] if isinstance(out, list) else out
                pred = r["pred_occupancy"] if isinstance(r, dict) else r
                pred = np.asarray(pred.cpu() if torch.is_tensor(pred) else pred).astype(np.uint8)
                send_msg(c, {"order": i, "pred": pred, "prep_ms": (t1-t0)*1e3, "infer_ms": (t2-t1)*1e3,
                             "gpu_peak_mb": torch.cuda.max_memory_allocated()/2**20, "pid": os.getpid(),
                             "imgs_from_ros": stats["ros"], "imgs_from_disk": stats["disk"]})
                stats["ros"] = stats["disk"] = 0
        except ConnectionError:
            print("client left", flush=True)

if __name__ == "__main__":
    main()
