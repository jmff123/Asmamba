class_weight = [
    1.0,
    12.0,
    2.0,
]
crop_size = (
    256,
    256,
)
custom_hooks = [
    dict(filename='training_metrics.csv', interval=50, type='CSVLoggerHook'),
    dict(
        log_interval=1000,
        num_classes=3,
        type='DCBSComplementHook',
        update_interval=1000),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmseg.datasets.samplers.dcbs_complement_sampler',
        'mmseg.engine.hooks.dcbs_complement_hook',
        'mmseg.models.decode_heads.uper_head_dict',
        'mmseg.models.utils.dynamic_dictionary',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        256,
        256,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data'
dataset_type = 'CustomThreeClassOversampleDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=5,
        rule='greater',
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, interval=500, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    inference_class_bias=[
        -2.8,
        -1.7,
        7.0,
    ],
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=320,
        in_index=2,
        loss_decode=[
            dict(
                activate=True,
                loss_weight=2.0,
                type='DiceLoss',
                use_sigmoid=False),
            dict(
                alpha=0.25,
                class_weight=[
                    1.0,
                    12.0,
                    2.0,
                ],
                gamma=2.5,
                loss_weight=1.2,
                type='FocalLoss',
                use_sigmoid=True),
            dict(
                class_weight=[
                    1.0,
                    12.0,
                    2.0,
                ],
                loss_type='multi_class',
                loss_weight=1.0,
                per_image=True,
                type='LovaszLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=3,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        depths=[
            3,
            4,
            6,
            3,
        ],
        drop_path_rate=0.15,
        embed_dims=[
            64,
            128,
            320,
            448,
        ],
        in_chans=3,
        mlp_ratios=[
            8,
            8,
            4,
            4,
        ],
        num_classes=1000,
        num_stages=4,
        sr_ratios=[
            4,
            2,
            1,
            1,
        ],
        stem_hidden_dim=32,
        token_label=True,
        type='Asmamba',
        use_4d_ssm=False,
        use_8d_ssm=False,
        use_adaptive_ssm=False,
        use_glss=False,
        use_sa_ssm=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dict_embed_dim=256,
        dict_num_heads=8,
        dict_query_ratio=4,
        dropout_ratio=0.1,
        in_channels=[
            64,
            128,
            320,
            448,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(
                activate=True,
                loss_weight=1.0,
                naive_dice=False,
                type='DiceLoss',
                use_sigmoid=False),
            dict(
                alpha=0.25,
                class_weight=[
                    1.0,
                    12.0,
                    2.0,
                ],
                gamma=2.5,
                loss_weight=1.0,
                type='FocalLoss',
                use_sigmoid=True),
            dict(
                class_weight=[
                    1.0,
                    12.0,
                    2.0,
                ],
                loss_type='multi_class',
                loss_weight=1.0,
                per_image=True,
                type='LovaszLoss'),
            dict(
                alpha=0.3,
                beta=0.7,
                class_weight=[
                    1.0,
                    12.0,
                    2.0,
                ],
                loss_weight=1.0,
                type='TverskyLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=3,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        relation_init='identity',
        total_iters=90000,
        type='UPerHeadDictEnhanced',
        use_adaptive_weights=True,
        use_contrastive_loss=True,
        use_enhanced_dict=True,
        use_relation_matrix=True,
        weight_strategy='hybrid'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.00018, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone': dict(lr_mult=0.1),
            'cls_token': dict(decay_mult=0.0),
            'enhanced_dict.dict_embeddings': dict(lr_mult=1.0),
            'enhanced_dict.relation_matrix': dict(lr_mult=1.0),
            'norm': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.0),
            'sa_ssm.adaptive_gate': dict(lr_mult=1.0),
            'sa_ssm.snake_mamba': dict(lr_mult=0.5)
        })),
    type='OptimWrapper')
optimizer = None
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=5063, start_factor=1e-06,
        type='LinearLR'),
    dict(
        T_max=84937,
        begin=5063,
        by_epoch=False,
        end=90000,
        eta_min=1e-07,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img/val', seg_map_path='mask/val'),
        data_root='data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CustomThreeClassOversampleDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    format_only=False,
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ],
    keep_results=True,
    output_dir=None,
    prefix=None,
    type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        256,
        256,
    ), type='Resize'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=80000, type='IterBasedTrainLoop', val_interval=2000)
train_dataloader = dict(
    batch_size=14,
    dataset=dict(
        data_prefix=dict(img_path='img', seg_map_path='mask'),
        data_root='data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.8,
                    1.2,
                ),
                scale=(
                    256,
                    256,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    256,
                    256,
                ), type='RandomCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(direction='vertical', prob=0.3, type='RandomFlip'),
            dict(
                brightness_delta=25,
                contrast_range=(
                    0.6,
                    1.4,
                ),
                hue_delta=15,
                saturation_range=(
                    0.6,
                    1.4,
                ),
                type='PhotoMetricDistortion'),
            dict(
                degree=(
                    -5,
                    5,
                ),
                pad_val=0,
                prob=0.3,
                seg_pad_val=255,
                type='RandomRotate'),
            dict(type='PackSegInputs'),
        ],
        type='CustomThreeClassDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(
        alpha_0=0.18,
        complement_start_iter=8000,
        complement_weight=0.7,
        high_precision_threshold=0.65,
        num_classes=3,
        reference_classes=[
            0,
            2,
        ],
        shuffle=True,
        target_class=1,
        total_iters=90000,
        type='IterBasedDCBSComplementSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(reduce_zero_label=False, type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.8,
            1.2,
        ),
        scale=(
            256,
            256,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        256,
        256,
    ), type='RandomCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(direction='vertical', prob=0.3, type='RandomFlip'),
    dict(
        brightness_delta=25,
        contrast_range=(
            0.6,
            1.4,
        ),
        hue_delta=15,
        saturation_range=(
            0.6,
            1.4,
        ),
        type='PhotoMetricDistortion'),
    dict(
        degree=(
            -5,
            5,
        ),
        pad_val=0,
        prob=0.3,
        seg_pad_val=255,
        type='RandomRotate'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='img/val', seg_map_path='mask/val'),
        data_root='data',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='Resize'),
            dict(reduce_zero_label=False, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='CustomThreeClassOversampleDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    format_only=False,
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ],
    keep_results=True,
    output_dir=None,
    prefix=None,
    type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'output/Asmamba_sa_ssm_upernet_dict_enhanced'
