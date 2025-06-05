_base_ = [
    'configs/_base_/datasets/cityscapes_1024x1024.py',
    'configs/_base_/default_runtime.py',
]

# The class_weight is borrowed from https://github.com/openseg-group/OCNet.pytorch/issues/14 # noqa
# Licensed under the MIT License
class_weight = [
    0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786,
    1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529,
    1.0507
]

crop_size = (1024, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DFDRNetV1',
        in_channels=3,
        channels=32,
        ppm_channels=128,
        norm_cfg=norm_cfg,
        align_corners=False
        #,
        # 关闭预训练
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
        ),
    decode_head=dict(
        type='DFDRHead',
        in_channels=32 * 4,
        channels=64,
        dropout_ratio=0.,
        num_classes=19,
        align_corners=False,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=0.4),
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=0.6),
        ]),


    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

train_dataloader = dict(batch_size=12, num_workers=4)
#初始(6,4)

iters = 120_000
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=optimizer,
                     clip_grad=None
                     )
# learning policy
param_scheduler = [
        # 主学习率调度器
        dict(
            type='PolyLR',
            eta_min=0,
            power=0.9,
            begin=0,
            end=iters,
            by_epoch=False)
    ]

# warmup 策略
# warm_up_iter=int(0.1*iters)
# param_scheduler = [
#         # # 线性学习率预热调度器
#         dict(type='LinearLR',
#              start_factor=0.01,
#              by_epoch=False,  # 按迭代更新学习率
#              begin=0,
#              end=warm_up_iter),  # 预热前 warm_up_iter 次迭代
#         # 主学习率调度器
#         dict(
#             type='PolyLR',
#             eta_min=0,
#             power=0.9,
#             begin=warm_up_iter,
#             end=iters,
#             by_epoch=False)
#     ]


# training schedule for 120k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=iters, val_interval=iters // 10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=iters // 10),
    # checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=iters // 10,
    #                 save_best='auto', published_keys=['meta', 'state_dict']),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# visualization=dict(type='SegVisualizationHook', draw=True, interval=1))

randomness = dict(seed=304)
