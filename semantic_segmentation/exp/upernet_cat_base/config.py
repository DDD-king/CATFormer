_base_ = [
    '../../configs/_base_/models/upernet_cat.py', 
    '../../configs/_base_/datasets/ade20k.py',
    '../../configs/_base_/default_runtime.py', 
    '../../configs/_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        type='CATFormer',
        depths=[4, 5, 12, 4],
        dims=[64, 128, 320, 512],
        num_heads=[2, 4, 8, 16],
        kernel_sizes=[5, 5, 1, 1],
        drop_path_rate=0.2,
        init_cfg=dict(type='Pretrained', checkpoint='/home/u000011200011/data/MMseg/mmsegmentation-master-0.30.0/pretrained/cat/base/best.pth')
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=320,
        num_classes=150
    ))

# we use 4 gpu in mmsegmentation

# AdamW optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
)
