# Zero-shot test of the nuScenes-trained ALOcc-3D (256x704, 16-frame history) on
# Occ3D-Waymo validation. Same model and checkpoint as the nuScenes baseline;
# only the test data changes. The dataset adapter is project/tools/waymo_zero_shot.py
# (project/tools must be on PYTHONPATH).
#
# Images: same test-time processing as nuScenes. Each image is resized to width
# 704 and the bottom 256 rows are kept (crop_h = 0). Waymo images are not padded
# to one size first, so the shorter side cameras are cropped from their own
# bottom edge rather than from black padding.
#
# History-reset variant: add --cfg-options data.test.sequences_split_num=all
# (every frame starts a new sequence, so no temporal history is carried over).
import os
# _base_ = ['/home/jihoon/work/ALOcc/configs/alocc/alocc_3d_r50_256x704_bevdet_preatrain_16f.py']
_base_ = ['../../configs/alocc/alocc_3d_r50_256x704_bevdet_preatrain_16f.py']
custom_imports = dict(imports=['waymo_zero_shot'], allow_failed_imports=False)

# making paths configurable:
# waymo_root = '/home/jihoon/data/waymo_alocc/'
# occ3d_waymo = '/home/jihoon/data/occ3d_waymo/'
waymo_root = os.environ.get('WAYMO_ROOT', './data/waymo_alocc/')
occ3d_waymo = os.environ.get('OCC3D_WAYMO_ROOT', './data/occ3d_waymo/')

waymo_data_config = {
    'cams': ['CAM_SIDE_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_SIDE_RIGHT'],
    'Ncams': 5,
    'input_size': (256, 704),
    'src_size': (1280, 1920),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

waymo_test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(type='PrepareImageInputs', data_config=waymo_data_config, sequential=True),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(rot_lim=(-0, 0), scale_lim=(1., 1.), flip_dx_ratio=0.5, flip_dy_ratio=0.5),
                classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                         'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'],
                is_train=False),
            dict(type='DefaultFormatBundle3D', class_names=[], with_label=False),
            dict(type='Collect3D', keys=['img_inputs', 'aux_cam_params', 'adj_aux_cam_params'])
        ])
]

data = dict(
    test=dict(
        _delete_=True,
        type='WaymoOccZeroShot',
        data_root=waymo_root,
        # ann_file=waymo_root + 'waymo_infos_val_jpg.pkl',
        ann_file=os.path.join(waymo_root, 'waymo_infos_val_jpg.pkl'),
        split='training',
        # pose_file=occ3d_waymo + 'cam_infos_vali.pkl',
        pose_file=os.path.join(occ3d_waymo, 'cam_infos_vali.pkl'),
        # occupancy_path=occ3d_waymo + 'validation',
        occupancy_path=os.path.join(occ3d_waymo, 'validation'),
        pipeline=waymo_test_pipeline,
        modality=dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False),
        test_mode=True,
        num_views=5,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        use_sequence_group_flag=True,
        sequences_split_num=1,
        stereo=True,
        multi_adj_frame_id_cfg=(1, 1, 1),
        frame_interval=5,
        scene_range=None))
