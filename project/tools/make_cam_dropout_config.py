"""Write a copy of an ALOcc config with some cameras removed.

Usage: python make_cam_dropout_config.py BASE_CONFIG OUT_CONFIG CAM [CAM ...]
"""
import sys

from mmcv import Config


def drop_cams(node, dropped):
    if isinstance(node, dict):
        if 'cams' in node and 'Ncams' in node:
            node['cams'] = [c for c in node['cams'] if c not in dropped]
            node['Ncams'] = len(node['cams'])
        for v in node.values():
            drop_cams(v, dropped)
    elif isinstance(node, list):
        for v in node:
            drop_cams(v, dropped)


base, out, *dropped = sys.argv[1:]
cfg = Config.fromfile(base)
drop_cams(cfg._cfg_dict, set(dropped))
cfg.dump(out)
print('cams:', cfg.data_config['cams'])
