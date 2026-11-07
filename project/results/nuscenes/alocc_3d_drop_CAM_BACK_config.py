point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ],
            Ncams=5,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0),
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=dict(
            rot_lim=(0, 0),
            scale_lim=(1.0, 1.0),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        point_with_semantic=True,
        file_client_args=dict(backend='disk')),
    dict(
        type='PointToMultiViewDepth',
        downsample=1,
        grid_config=dict(
            x=[-40, 40, 0.4],
            y=[-40, 40, 0.4],
            z=[-1, 5.4, 0.4],
            depth=[1.0, 45.0, 0.5]),
        load_semantic_map=True,
        fix_void=True),
    dict(
        type='LoadOccupancy',
        ignore_nonvisible=True,
        fix_void=True,
        occupancy_path='data/nuscenes/gts'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'gt_occupancy', 'gt_depth', 'aux_cam_params',
            'adj_aux_cam_params', 'gt_semantic_map'
        ])
]
test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=5,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                sequential=True),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(0, 0),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=[
                    'points', 'img_inputs', 'aux_cam_params',
                    'adj_aux_cam_params'
                ])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_train.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                is_train=True,
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=5,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                sequential=True),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(0, 0),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                point_with_semantic=True,
                file_client_args=dict(backend='disk')),
            dict(
                type='PointToMultiViewDepth',
                downsample=1,
                grid_config=dict(
                    x=[-40, 40, 0.4],
                    y=[-40, 40, 0.4],
                    z=[-1, 5.4, 0.4],
                    depth=[1.0, 45.0, 0.5]),
                load_semantic_map=True,
                fix_void=True),
            dict(
                type='LoadOccupancy',
                ignore_nonvisible=True,
                fix_void=True,
                occupancy_path='data/nuscenes/gts'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'img_inputs', 'gt_occupancy', 'gt_depth', 'aux_cam_params',
                    'adj_aux_cam_params', 'gt_semantic_map'
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True,
        img_info_prototype='bevdet4d',
        sequences_split_num=2,
        use_sequence_group_flag=True,
        filter_empty_gt=False,
        stereo=True,
        multi_adj_frame_id_cfg=(1, 1, 1)),
    val=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='CustomDistMultiScaleFlipAug3D',
                tta=False,
                transforms=[
                    dict(
                        type='PrepareImageInputs',
                        data_config=dict(
                            cams=[
                                'CAM_FRONT_LEFT', 'CAM_FRONT',
                                'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
                                'CAM_BACK_RIGHT'
                            ],
                            Ncams=5,
                            input_size=(256, 704),
                            src_size=(900, 1600),
                            resize=(-0.06, 0.11),
                            rot=(-5.4, 5.4),
                            flip=True,
                            crop_h=(0.0, 0.0),
                            resize_test=0.0),
                        sequential=True),
                    dict(
                        type='LoadAnnotationsBEVDepth',
                        bda_aug_conf=dict(
                            rot_lim=(0, 0),
                            scale_lim=(1.0, 1.0),
                            flip_dx_ratio=0.5,
                            flip_dy_ratio=0.5),
                        classes=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        is_train=False),
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=5,
                        use_dim=5,
                        file_client_args=dict(backend='disk')),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'points', 'img_inputs', 'aux_cam_params',
                            'adj_aux_cam_params'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        sequences_split_num=1,
        img_info_prototype='bevdet4d',
        occupancy_path='data/nuscenes/gts',
        use_sequence_group_flag=True,
        stereo=True,
        multi_adj_frame_id_cfg=(1, 1, 1)),
    test=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
        pipeline=[
            dict(
                type='CustomDistMultiScaleFlipAug3D',
                tta=False,
                transforms=[
                    dict(
                        type='PrepareImageInputs',
                        data_config=dict(
                            cams=[
                                'CAM_FRONT_LEFT', 'CAM_FRONT',
                                'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
                                'CAM_BACK_RIGHT'
                            ],
                            Ncams=5,
                            input_size=(256, 704),
                            src_size=(900, 1600),
                            resize=(-0.06, 0.11),
                            rot=(-5.4, 5.4),
                            flip=True,
                            crop_h=(0.0, 0.0),
                            resize_test=0.0),
                        sequential=True),
                    dict(
                        type='LoadAnnotationsBEVDepth',
                        bda_aug_conf=dict(
                            rot_lim=(0, 0),
                            scale_lim=(1.0, 1.0),
                            flip_dx_ratio=0.5,
                            flip_dy_ratio=0.5),
                        classes=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        is_train=False),
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=5,
                        use_dim=5,
                        file_client_args=dict(backend='disk')),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'points', 'img_inputs', 'aux_cam_params',
                            'adj_aux_cam_params'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        sequences_split_num=1,
        img_info_prototype='bevdet4d',
        occupancy_path='data/nuscenes/gts',
        use_sequence_group_flag=True,
        stereo=True,
        multi_adj_frame_id_cfg=(1, 1, 1)))
evaluation = dict(
    interval=96060,
    pipeline=[
        dict(
            type='CustomDistMultiScaleFlipAug3D',
            tta=False,
            transforms=[
                dict(
                    type='PrepareImageInputs',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=5,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.0),
                    sequential=True),
                dict(
                    type='LoadAnnotationsBEVDepth',
                    bda_aug_conf=dict(
                        rot_lim=(0, 0),
                        scale_lim=(1.0, 1.0),
                        flip_dx_ratio=0.5,
                        flip_dy_ratio=0.5),
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    is_train=False),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=[
                        'points', 'img_inputs', 'aux_cam_params',
                        'adj_aux_cam_params'
                    ])
            ])
    ])
checkpoint_config = dict(interval=8005)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = 'ckpts/pretrain/bevdet-r50-4d-stereo-cbgs.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
num_gpus = 8
samples_per_gpu = 2
num_iters_per_epoch = 8005
num_epochs = 12
checkpoint_epoch_interval = 1
use_custom_eval_hook = True
train_sequences_split_num = 2
test_sequences_split_num = 1
filter_empty_gt = False
history_cat_num = 16
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
    ],
    Ncams=5,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
bda_aug_conf = dict(
    rot_lim=(0, 0), scale_lim=(1.0, 1.0), flip_dx_ratio=0.5, flip_dy_ratio=0.5)
use_checkpoint = True
sync_bn = True
grid_config = dict(
    x=[-40, 40, 0.4],
    y=[-40, 40, 0.4],
    z=[-1, 5.4, 0.4],
    depth=[1.0, 45.0, 0.5])
depth_channels = 88
numC_Trans = 32
_dim_ = 256
empty_idx = 18
num_cls = 19
fix_void = True
voxel_out_channel = 48
occ_encoder_channels = [32, 64, 128, 256]
not_use_history = False
use_sequence_group_flag = True
depth_stereo = True
multi_adj_frame_id_cfg = (1, 1, 1)
load_point_semantic = True
img_seg_weight = 0.1
sem_sup_prototype = True
depth2occ_intra = True
length = 22
depth2occ_inter = True
soft_filling = True
use_depth_supervision = True
geometry_denoise = True
depth_loss_ce = True
wo_assign = True
use_mask_net2 = True
model = dict(
    type='ALOCC',
    use_depth_supervision=True,
    fix_void=True,
    history_cat_num=16,
    single_bev_num_channels=32,
    not_use_history=False,
    depth_stereo=True,
    grid_config=dict(
        x=[-40, 40, 0.4],
        y=[-40, 40, 0.4],
        z=[-1, 5.4, 0.4],
        depth=[1.0, 45.0, 0.5]),
    downsample=16,
    img_seg_weight=0.1,
    depth_loss_ce=True,
    depth2occ_intra=True,
    sem_sup_prototype=True,
    use_mask_net2=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        with_cp=True,
        out_ids=[0]),
    depth_net=dict(
        type='CM_DepthNet',
        in_channels=256,
        context_channels=32,
        downsample=16,
        grid_config=dict(
            x=[-40, 40, 0.4],
            y=[-40, 40, 0.4],
            z=[-1, 5.4, 0.4],
            depth=[1.0, 45.0, 0.5]),
        depth_channels=88,
        with_cp=True,
        loss_depth_weight=1.0,
        use_dcn=False,
        stereo=True,
        input_size=(256, 704),
        bias=5.0,
        mid_channels=256,
        aspp_mid_channels=96,
        depth2occ_intra=True,
        length=22,
        depth2occ_inter=True,
        geometry_denoise=True),
    view_transformer=dict(
        type='LSSViewTransformerFunction',
        grid_config=dict(
            x=[-40, 40, 0.4],
            y=[-40, 40, 0.4],
            z=[-1, 5.4, 0.4],
            depth=[1.0, 45.0, 0.5]),
        input_size=(256, 704),
        soft_filling=True,
        depth2occ_inter=True,
        downsample=16),
    pre_process=dict(
        type='CustomResNet3D',
        numC_input=32,
        with_cp=True,
        num_layer=[1],
        num_channels=[32],
        stride=[1],
        backbone_output_ids=[0]),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        numC_input=32,
        num_layer=[1, 2, 4, 4],
        with_cp=True,
        num_channels=[32, 64, 128, 256],
        stride=[1, 2, 2, 2],
        backbone_output_ids=[0, 1, 2, 3]),
    img_bev_encoder_neck=dict(
        type='LSSFPN3D2', in_channels=480, out_channels=48),
    alocc_head=dict(
        type='ALOccHead',
        feat_channels=48,
        out_channels=48,
        num_queries=19,
        num_occupancy_classes=19,
        sample_weight_gamma=0.25,
        num_transformer_feat_level=0,
        wo_assign=True,
        mask_embed2=True,
        out_channels_embed2=32,
        num_points_img=1056,
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=0,
            transformerlayers=None,
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1
            ]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=25088,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=None,
            sampler=None)),
    pts_bbox_head=None)
occupancy_path = 'data/nuscenes/gts'
share_data_config = dict(
    type='NuScenesDataset',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet4d',
    occupancy_path='data/nuscenes/gts',
    use_sequence_group_flag=True,
    stereo=True,
    multi_adj_frame_id_cfg=(1, 1, 1))
test_data_config = dict(
    pipeline=[
        dict(
            type='CustomDistMultiScaleFlipAug3D',
            tta=False,
            transforms=[
                dict(
                    type='PrepareImageInputs',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=5,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.0),
                    sequential=True),
                dict(
                    type='LoadAnnotationsBEVDepth',
                    bda_aug_conf=dict(
                        rot_lim=(0, 0),
                        scale_lim=(1.0, 1.0),
                        flip_dx_ratio=0.5,
                        flip_dy_ratio=0.5),
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    is_train=False),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=[
                        'points', 'img_inputs', 'aux_cam_params',
                        'adj_aux_cam_params'
                    ])
            ])
    ],
    sequences_split_num=1,
    ann_file='data/nuscenes/bevdetv2-nuscenes_infos_val.pkl',
    type='NuScenesDataset',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet4d',
    occupancy_path='data/nuscenes/gts',
    use_sequence_group_flag=True,
    stereo=True,
    multi_adj_frame_id_cfg=(1, 1, 1))
key = 'test'
lr = 0.0002
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[96060])
runner = dict(type='IterBasedRunner', max_iters=96060)
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=8005),
    dict(type='SequentialControlHook', temporal_start_iter=16010),
    dict(
        type='FusionRateControlDepthHook',
        temporal_start_iter=0,
        temporal_end_iter=48030)
]
