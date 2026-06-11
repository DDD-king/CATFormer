_base_ = [
    '../../configs/_base_/models/mask_rcnn_cat_fpn.py',
    '../../configs/_base_/datasets/coco_instance.py',
    '../../configs/_base_/schedules/schedule_1x.py', 
    '../../configs/_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        depths=[5, 8, 20, 7],
        dims=[64, 128, 320, 512],
        num_heads=[2, 4, 8, 16],
        kernel_sizes=[5, 5, 1, 1],
        drop_path_rate=0.3,
        init_cfg=dict(type='Pretrained', checkpoint='/home/u000011200011/data/MMdet/mmdetection-master-2.25.1/pretrained/cat/large/best.pth')
    ),
    neck=dict(in_channels=[64, 128, 320, 512]))

# we use 8 gpu in mmsegmentation

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[8, 11])
#runner = dict(type='EpochBasedRunner', max_epochs=12)
#
## Mixed precision
## do not use mmdet version fp16
#fp16 = None
#optimizer_config = dict(
#    type="Fp16OptimizerHook",
#    grad_clip=None,
#    coalesce=True,
#    bucket_size_mb=-1
#)

fp16 = dict()
###########################################################################################################

# place holder for new verison mmdet compatiability
resume_from=None

# custom
checkpoint_config = dict(max_keep_ckpts=1)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)