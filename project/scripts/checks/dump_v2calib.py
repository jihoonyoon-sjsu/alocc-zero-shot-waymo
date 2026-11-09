"""Save one scene's Waymo v2 camera_calibration as a dict (for envs without pyarrow)."""
import glob, os, sys
import numpy as np
import pyarrow.parquet as pq
scene = int(sys.argv[1]); V2 = '/home/jihoon/data/waymo_v2/validation'
seg = sorted(os.path.basename(p)[:-8] for p in glob.glob(f'{V2}/camera_image/*.parquet'))[scene]
os.makedirs('/home/jihoon/work/results/figures', exist_ok=True)
np.save(f'/home/jihoon/work/results/figures/v2calib_{scene:03d}.npy', pq.read_table(f'{V2}/camera_calibration/{seg}.parquet').to_pydict())
