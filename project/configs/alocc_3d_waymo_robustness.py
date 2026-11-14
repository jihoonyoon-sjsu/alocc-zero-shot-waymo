# Waymo robustness experiments
# Inherits the original zero-shot configuration.

# _base_ = ['./alocc_3d_waymo_zero_shot.py']
_base_ = ['./alocc_3d_waymo_portable.py']
# _base_ = [
#     '../../configs/alocc/alocc_3d_r50_256x704_bevdet_preatrain_16f.py'
# ]

waymo_data_config = {
    'cams': ['CAM_SIDE_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT',
             'CAM_FRONT_RIGHT', 'CAM_SIDE_RIGHT'],
    'Ncams': 5,
    'input_size': (256, 704),
    'src_size': (1280, 1920),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# Register our custom image preprocessing class.
custom_imports = dict(
    imports=[
        'waymo_zero_shot',
        'project.robustness.robustness_pipeline',
    ],
    allow_failed_imports=False,
)

# Copy the original pipeline so it remains unchanged.
robust_test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(
                type='RobustPrepareImageInputs',
                data_config=waymo_data_config,
                sequential=True,
                corruption=None,
                severity=1,
                crop_shift=50,
            ),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(0, 0),
                    scale_lim=(1., 1.),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5,
                ),
                classes=[
                    'car', 'truck', 'construction_vehicle',
                    'bus', 'trailer', 'barrier', 'motorcycle',
                    'bicycle', 'pedestrian', 'traffic_cone',
                ],
                is_train=False,
            ),
            dict(type='DefaultFormatBundle3D', class_names=[], with_label=False),
            dict(
                type='Collect3D',
                keys=['img_inputs', 'aux_cam_params', 'adj_aux_cam_params'],
            ),
        ],
    )
]

data = dict(test=dict(pipeline=robust_test_pipeline))