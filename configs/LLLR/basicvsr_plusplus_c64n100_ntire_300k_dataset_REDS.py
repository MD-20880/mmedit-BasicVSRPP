exp_name = 'basicvsr_plusplus_c64n100_ntire_300k_datasetREDS'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlus',
        mid_channels=64,
        num_blocks=10,
        max_residue_magnitude = 10,
        is_low_res_input=True,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth',
        cpu_cache_length = 100),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = dict(fix_iter=-1)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0, convert_to='y')

# dataset settings
train_dataset_type = 'SRREDSMultipleGTDataset'
val_dataset_type = 'SRREDSMultipleGTDataset'    
test_dataset_type = 'SRREDSMultipleGTDataset'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

val_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

test_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='MirrorSequence', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),  # 4 gpus
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1),

    # train
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=train_dataset_type,
            lq_folder='D:\\Dataset\\dataset_REDS\\Synthetic4x\\train',
            gt_folder='D:\\Dataset\\dataset_REDS\\groundtruth\\train',
            num_input_frames = 5,
            pipeline=train_pipeline,
            scale=2,
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
            lq_folder='D:\\Dataset\\dataset_REDS\\Synthetic4x\\train',
            gt_folder='D:\\Dataset\\dataset_REDS\\groundtruth\\train',
            num_input_frames = 5,
            pipeline=val_pipeline,
            scale=2,
            test_mode=False),
    #test
    test=dict(
       type=test_dataset_type,
            lq_folder='D:\\Dataset\\dataset_REDS\\Synthetic4x\\test',
            gt_folder='D:\\Dataset\\dataset_REDS\\groundtruth\\test',
            num_input_frames = 5,
            pipeline=test_pipeline,
            scale=2,
            test_mode=True),
 )

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99),
        paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)})))

# learning policy
total_iters = 3000000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[300000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=50000, save_image=False)
log_config = dict(
    interval=1000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
