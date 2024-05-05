data_dir = 'DATASET_PATH'
work_dir = 'CACHE_DIR'

_base_ = ['./default_runtime.py']

custom_classes = [i+1 for i in range(34)]
for i in [1, 4, 7, 12, 20, 30]:
    custom_classes.remove(i)
num_classes = len(custom_classes) + 1

url = (
    'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/'
    'vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth')

model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=4,
        norm_cfg=dict(type='LN', eps=1e-6),
        drop_path_rate=0.2,
        use_mean_pooling=False,
        return_feat_map=True),
    roi_head=dict(
        type='HHIRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadHHI',
            in_channels=768, # 2304 * 1
            num_classes=num_classes,
            multilabel=False,
            dropout_ratio=0.1,
            use_spatial=True,
            use_attention=True)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        _scope_='mmaction',
        mean=[151.24, 131.09, 130.70],
        std=[67.88, 58.51, 54.34],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerHHI',
                pos_iou_thr=.7,
                neg_iou_thr=.7,
                min_pos_iou=.7),
            sampler=dict(
                type='HHIRandomSampler',
                num=32,
                pos_fraction=1.,
                neg_pos_ub=10.,
                add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False,
            nms=dict( # Config of NMS
                type='nms',  # Type of NMS
                iou_threshold=0.85 # NMS threshold
                ),)),
    test_cfg=dict(rcnn=dict(action_thr=-0.02))) # single label setting for SportsHHI, actions with scores larger than 0.02

dataset_type = 'SportsHHIDataset'
data_root = f'{data_dir}/frames'
anno_root = f'{data_dir}/annotations'

ann_file_train = f'{anno_root}/sports_train_v1.csv'
ann_file_val = f'{anno_root}/sports_val_v1.csv'

label_file = f'{anno_root}/sports_action_list.pbtxt'

proposal_file_train = f'{anno_root}/sports_dense_proposals_train.pkl'
proposal_file_val = f'{anno_root}/sports_dense_proposals_val.pkl'

img_norm_cfg = dict(
    mean=[151.24, 131.09, 130.70], std=[67.88, 58.51, 54.34], to_bgr=False)

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=4, frame_interval=2),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs', meta_keys=['scores', 'p1_ids', 'p2_ids', 'gt_interactions', 'img_shape', 'img_key', 'video_id', 'timestamp'])
]
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=4, frame_interval=2, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs', meta_keys=['scores', 'p1_ids', 'p2_ids', 'gt_interactions', 'img_shape', 'img_key', 'video_id', 'timestamp'])
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=None,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root),
        num_classes=num_classes,
        custom_classes=custom_classes))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=None,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        num_classes=num_classes,
        custom_classes=custom_classes,
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='HHIMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=None,
    num_classes=num_classes,
    custom_classes=custom_classes)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=15,
        eta_min=0,
        by_epoch=True,
        begin=5,
        end=20,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=4e-5, weight_decay=0.05),
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.75,
        'decay_type': 'layer_wise',
        'num_layers': 12
    },
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))

auto_scale_lr = dict(enable=False, base_batch_size=32)
