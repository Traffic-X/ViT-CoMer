_base_ = [
    'co_dino_5scale_r50_1x_coco.py'
]

load_from = 'o365_backbone_codetr_head.pth'
# load_from = 'work_dirs/co_dino_5scale_vitcomer_16e_o365tococo/epoch_13.pth'

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='BEiTComer2',
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        use_abs_pos_emb=False,
        use_rel_pos_bias=True,
        init_values=1e-6,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        with_cp=True,
        use_injector=[True, True, True, True], 
        use_extractor=[True, True, True, True],
        cnn_feature_interaction=[True, True, True, True],
        window_attn=[True, True, True, True, True, True,
                     True, True, True, True, True, True,
                     True, True, True, True, True, True,
                     True, True, True, True, True, True],
        window_size=[14, 14, 14, 14, 14, 56,
                     14, 14, 14, 14, 14, 56,
                     14, 14, 14, 14, 14, 56,
                     14, 14, 14, 14, 14, 56],
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        pretrained=None),
    # neck=dict(in_channels=[1024, 1024, 1024, 1024]),
    neck=dict(        
        type='SFP',
        in_channels=[1024],        
        out_channels=256,
        num_outs=5,
        use_p2=True,
        use_act_checkpoint=False),
    query_head=dict(
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=0.4),
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=500)),
        transformer=dict(
            encoder=dict(
                # number of layers that use checkpoint
                with_cp=6))))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 2176), (512, 2176), (544, 2176), (576, 2176),
                               (608, 2176), (640, 2176), (672, 2176), (704, 2176),
                               (736, 2176), (768, 2176), (800, 2176), (832, 2176),
                               (864, 2176), (896, 2176), (928, 2176), (960, 2176),
                               (992, 2176), (1024, 2176), (1056, 2176), (1088, 2176),
                               (1120, 2176), (1152, 2176), (1184, 2176), (1216, 2176),
                               (1248, 2176), (1280, 2176), (1312, 2176), (1344, 2176),
                               (1376, 2176), (1408, 2176), (1440, 2176), (1472, 2176),
                               (1504, 2176), (1536, 2176), (1568, 2176), (1600, 2176)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 2176), (512, 2176), (544, 2176), (576, 2176),
                               (608, 2176), (640, 2176), (672, 2176), (704, 2176),
                               (736, 2176), (768, 2176), (800, 2176), (832, 2176),
                               (864, 2176), (896, 2176), (928, 2176), (960, 2176),
                               (992, 2176), (1024, 2176), (1056, 2176), (1088, 2176),
                               (1120, 2176), (1152, 2176), (1184, 2176), (1216, 2176),
                               (1248, 2176), (1280, 2176), (1312, 2176), (1344, 2176),
                               (1376, 2176), (1408, 2176), (1440, 2176), (1472, 2176),
                               (1504, 2176), (1536, 2176), (1568, 2176), (1600, 2176)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2176, 1312),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(filter_empty_gt=False, pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.05,
    # custom_keys of sampling_offsets and reference_points in DeformDETR
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

# optimizer = dict(
#     _delete_=True, type='AdamW', lr=0.00005, weight_decay=0.05,
#     constructor='LayerDecayOptimizerConstructor',
#     paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.80))

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[8])
runner = dict(type='EpochBasedRunner', max_epochs=16)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)