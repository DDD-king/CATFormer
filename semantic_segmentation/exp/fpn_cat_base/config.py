_base_ = [
    '../../configs/_base_/models/fpn_cat.py',
    '../../configs/_base_/datasets/ade20k.py',
    '../../configs/_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        depths=[4, 5, 12, 4],
        dims=[64, 128, 320, 512],
        num_heads=[2, 4, 8, 16],
        kernel_sizes=[5, 5, 1, 1],
        drop_path_rate=0.2,
        init_cfg=dict(type='Pretrained', checkpoint='/home/u000011200011/data/MMseg/mmsegmentation-master-0.30.0/pretrained/cat/base/best.pth')
    ),
    neck=dict(in_channels=[64, 128, 320, 512]),
        decode_head=dict(num_classes=150)
    )

gpu_multiples=1  # we use 4 gpu in mmsegmentation

# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()

# learning policy
#lr_config = dict(
#    policy='CosineAnnealing',
#    warmup='linear',
#    warmup_iters=1000,
#    warmup_ratio=1.0 / 10,
#    min_lr_ratio=1e-8
#)
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU')
