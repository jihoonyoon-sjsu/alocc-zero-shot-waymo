import json, pickle, socket, struct, sys, time, os
import numpy as np
from PIL import Image
D = os.path.expanduser("~/work/ros_export/alocc_waymo_scene0")
frames = json.load(open(D + "/frames.json"))
ref = np.load(D + "/check/zero_shot_scene0.npz")["pred_occupancy"]
s = socket.socket(socket.AF_UNIX); s.connect("/tmp/alocc_server.sock")
def send(o):
    d = pickle.dumps(o, protocol=4); s.sendall(struct.pack("!Q", len(d)) + d)
def recv():
    def n(k):
        b = bytearray()
        while len(b) < k: b += s.recv(k - len(b))
        return bytes(b)
    return pickle.loads(n(struct.unpack("!Q", n(8))[0]))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
for f in frames[:N]:
    imgs = {}
    for cam, rel in f["images"].items():
        im = Image.open(D + "/" + rel).convert("RGB"); imgs[cam] = (im.height, im.width, im.tobytes())
    t = time.perf_counter(); send({"order": f["order"], "images": imgs}); r = recv(); rt = (time.perf_counter()-t)*1e3
    agree = (r["pred"] == ref[f["order"]]).mean()*100
    print(f"frame {f['order']:2d} agree {agree:6.2f}% infer {r['infer_ms']:6.1f}ms prep {r['prep_ms']:6.1f}ms roundtrip {rt:6.1f}ms gpu {r['gpu_peak_mb']:.0f}MB ros_imgs {r['imgs_from_ros']} disk_imgs {r['imgs_from_disk']}", flush=True)
send({"cmd": "quit"}); recv()
