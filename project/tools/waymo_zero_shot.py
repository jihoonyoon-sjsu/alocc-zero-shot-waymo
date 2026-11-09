"""Dataset adapter for running the nuScenes-trained ALOcc model on Occ3D-Waymo
(zero-shot). Builds on ALOcc's own WaymoDatasetOcc and changes only what the
nuScenes model needs:

1. 2 Hz like nuScenes keyframes: keep every 5th Waymo frame (Waymo is 10 Hz).
   ALOcc's load_interval cannot be used for this (its __len__ and index lookup
   do not agree when load_interval > 1).
2. Ego motion between kept frames: ALOcc computes it from the raw previous
   10 Hz frame; here it is computed from the previous kept frame.
3. Camera format: Occ3D-Waymo stores each camera in the Waymo camera frame
   (x forward, y left, z up) with a matching intrinsic matrix. The model's
   depth net reads fx, fy, cx, cy straight from the intrinsic matrix, so cameras
   are converted to the standard optical frame (x right, y down, z forward)
   with a standard K, as in nuScenes. The projection is unchanged.
4. Camera labels: slots 2 and 3 of cam_infos_vali.pkl are SIDE_LEFT and
   FRONT_RIGHT (checked against the Waymo v2 calibration); ALOcc labels them the
   other way round. Only the names change; images and poses already match.
5. Evaluation: predictions use nuScenes classes, so ALOcc's Waymo evaluation does
   not apply. Predictions are saved with --out and scored by the project scoring.

Also adds scene_range=[start, end) to run a subset of scenes (used for chunks).
"""
import numpy as np
import torch
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.waymo_occ import WaymoDatasetOcc

# Rotation from the Waymo camera frame to the optical camera frame.
WAYMO_TO_OPTICAL = np.array([[0., -1., 0.],
                             [0., 0., -1.],
                             [1., 0., 0.]])
# Occ3D cam_infos_vali.pkl slot order (checked against Waymo v2 calibration).
CAM_NAMES = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_SIDE_LEFT', 'CAM_FRONT_RIGHT', 'CAM_SIDE_RIGHT']


def scene_frame(info):
    idx = info['image']['image_idx'] % 1000000
    return idx // 1000, idx % 1000


def keep_frame(info, frame_interval=5, scene_range=None):
    scene, frame = scene_frame(info)
    in_range = scene_range is None or scene_range[0] <= scene < scene_range[1]
    return in_range and frame % frame_interval == 0


@DATASETS.register_module()
class WaymoOccZeroShot(WaymoDatasetOcc):

    def __init__(self, *args, frame_interval=5, scene_range=None, **kwargs):
        self.frame_interval = frame_interval
        self.scene_range = scene_range
        kwargs.setdefault('cam_names', CAM_NAMES)
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        infos = super().load_annotations(ann_file)
        infos = sorted(infos, key=lambda i: i['image']['image_idx'])
        return [i for i in infos if keep_frame(i, self.frame_interval, self.scene_range)]

    def __len__(self):
        return len(self.data_infos)

    def prepare_img_info(self):
        super().prepare_img_info()
        optical_to_waymo = np.eye(4)
        optical_to_waymo[:3, :3] = WAYMO_TO_OPTICAL.T
        for info in self.data_infos:
            for cam in info['cams'].values():
                K = np.asarray(cam['cam_intrinsic'], dtype=np.float64) @ WAYMO_TO_OPTICAL.T
                assert abs(K[0, 1]) < 1e-6 and abs(K[1, 0]) < 1e-6 and np.allclose(K[2], [0, 0, 1]), K
                cam['cam_intrinsic'] = K
                cam['cam2camego'] = np.asarray(cam['cam2camego'], dtype=np.float64) @ optical_to_waymo

    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        if self.use_sequence_group_flag and not input_dict['start_of_sequence']:
            prev = np.asarray(self.data_infos[index - 1]['lidarego2global'], dtype=np.float64)
            curr = np.asarray(self.data_infos[index]['lidarego2global'], dtype=np.float64)
            curr_to_prev = torch.from_numpy(np.linalg.inv(prev) @ curr).float()
            input_dict['curr_to_prev_lidar_rt'] = curr_to_prev
            input_dict['curr_to_prev_ego_rt'] = curr_to_prev
            input_dict['prev_lidar_to_global_rt'] = torch.from_numpy(prev).float()
        return input_dict

    def evaluate(self, results, **kwargs):
        print(f'WaymoOccZeroShot: {len(results)} results saved; scored separately.')
        return dict(num_results=len(results))
