# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='CATFormer',
        depths=[3, 3, 8, 4],
        dims=[64, 128, 320, 512],
        num_heads=[2, 4, 8, 16],
        kernel_sizes=[5, 5, 1, 1],
        drop_path_rate=0.1,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_after_stage=True,
        with_cp=True,
        init_cfg=dict(type='Pretrained', checkpoint='/mnt/guoqingbei/mmsegmentation-master-0.30.0/pretrained/small/best.pth')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
