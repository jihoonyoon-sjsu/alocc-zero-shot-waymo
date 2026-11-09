"""Extract Waymo v2 camera images into the layout ALOcc's WaymoDatasetOcc reads.

Layout: OUT/training/image_{k}/{image_idx}.jpg, where
  k         = Waymo camera_name - 1 (FRONT=0, FRONT_LEFT=1, FRONT_RIGHT=2,
              SIDE_LEFT=3, SIDE_RIGHT=4), the standard mmdet3d Waymo naming
  image_idx = 1000000 + scene_idx * 1000 + frame_idx (validation prefix 1)
  scene_idx = position of the segment in sorted segment names
  frame_idx = position of the frame in sorted timestamps
This matches Occ3D-Waymo's waymo_infos_val.pkl and cam_infos_vali.pkl
(checked: same frame counts and identical ego poses for every scene checked).

Images are written as the original JPEG bytes (no re-encoding). The infos file
names them .png, so a copy of waymo_infos_val.pkl with .jpg paths is written too.

Usage: python extract_waymo_images.py WAYMO_V2_VALIDATION_DIR OCC3D_WAYMO_DIR OUT_DIR
"""
import glob
import os
import pickle
import sys

import pyarrow.parquet as pq

v2_dir, occ_dir, out_dir = sys.argv[1:4]
cols = ['key.frame_timestamp_micros', 'key.camera_name', '[CameraImageComponent].image']

segs = sorted(os.path.basename(p)[:-8] for p in glob.glob(f'{v2_dir}/camera_image/*.parquet'))
assert len(segs) == 202, len(segs)
for k in range(5):
    os.makedirs(f'{out_dir}/training/image_{k}', exist_ok=True)

for scene_idx, seg in enumerate(segs):
    t = pq.read_table(f'{v2_dir}/camera_image/{seg}.parquet', columns=cols).to_pydict()
    stamps = sorted(set(t['key.frame_timestamp_micros']))
    frame_of = {s: i for i, s in enumerate(stamps)}
    for stamp, cam, img in zip(*(t[c] for c in cols)):
        idx = 1000000 + scene_idx * 1000 + frame_of[stamp]
        path = f'{out_dir}/training/image_{cam - 1}/{idx}.jpg'
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                f.write(img)
    print(f'{scene_idx + 1}/202 {seg}: {len(stamps)} frames', flush=True)

with open(f'{occ_dir}/waymo_infos_val.pkl', 'rb') as f:
    infos = pickle.load(f)
for info in infos:
    info['image']['image_path'] = info['image']['image_path'].replace('.png', '.jpg')
with open(f'{out_dir}/waymo_infos_val_jpg.pkl', 'wb') as f:
    pickle.dump(infos, f)
print('done')
