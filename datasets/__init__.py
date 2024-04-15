#  Authors: Zhaoshuo Li, Xingtong Liu, Francis X. Creighton, Russell H. Taylor, and Mathias Unberath
#
#  Copyright (c) 2020. Johns Hopkins University - All rights reserved.

import torch.utils.data as data

from datasets.scene_flow import SceneFlowDataset
from datasets.kitti1215_dataset import KITTIDataset
from datasets.eth3d import Eth3dDataset
from datasets.middlebury import MiddleBuryDataset
from datasets.spring import SpringDataset


def build_data_loader(opts):
    '''
    Build data loader

    :param opts: arg parser object
    :return: train, validation and test dataloaders
    '''
    if opts.dataset_path == '':
        raise ValueError(f'Dataset path cannot be empty.')
    else:
        dataset_dir = opts.dataset_path

    if opts.dataset == 'sceneflow':
        dataset_train = SceneFlowDataset(dataset_dir, "filenames/sceneflow_train.txt", opts.train_height, opts.train_width, training=True)
        dataset_validation = SceneFlowDataset(dataset_dir, "filenames/sceneflow_test.txt", opts.val_height, opts.val_width, training=False)
        dataset_test = SceneFlowDataset(dataset_dir, "filenames/sceneflow_test.txt", opts.val_height, opts.val_width, training=False)
    elif opts.dataset == 'kitti':
        dataset_train = KITTIDataset(dataset_dir, './filenames/kitti12_15_all.txt', opts.train_height,opts.train_width, mode='train')
        dataset_validation = KITTIDataset(dataset_dir, './filenames/kitti15_val3.txt', opts.val_height,
                                          opts.val_width, mode='eval')
        dataset_test = KITTIDataset(dataset_dir, './filenames/kitti12_test.txt', opts.val_height, opts.val_width, mode='test')
    elif opts.dataset == 'kitti12':
        dataset_train = None
        dataset_validation = None
        dataset_test = KITTIDataset(dataset_dir, './filenames/kitti12_test.txt', opts.val_height, opts.val_width, mode='test')
    elif opts.dataset == 'kitti15':
        dataset_train = None
        dataset_validation = None
        dataset_test = KITTIDataset(dataset_dir, './filenames/kitti15_test.txt', opts.val_height, opts.val_width, mode='test')
    elif opts.dataset == 'eth3d':
        dataset_train = Eth3dDataset(dataset_dir, './filenames/eth3d_train.txt', opts.train_height,opts.train_width, mode='train')
        dataset_validation = Eth3dDataset(dataset_dir, './filenames/eth3d_train.txt', opts.val_height, opts.val_width, mode='eval')
        dataset_test = Eth3dDataset(dataset_dir, './filenames/eth3d_test.txt', opts.val_height, opts.val_width, mode='test')
    elif opts.dataset == 'middlebury':
        dataset_train = MiddleBuryDataset(dataset_dir, opts.train_height,opts.train_width, mode='train')
        dataset_validation = MiddleBuryDataset(dataset_dir, opts.val_height, opts.val_width, mode='eval')
        dataset_test = MiddleBuryDataset(dataset_dir, opts.val_height, opts.val_width, mode='test')
    elif opts.dataset == 'spring':
        dataset_train = SpringDataset(dataset_dir, "filenames/springtrain_all.txt", opts.train_height, opts.train_width, mode='train')
        dataset_validation = SpringDataset(dataset_dir, "filenames/springval1.txt", opts.val_height, opts.val_width, mode='eval')
        dataset_test = SpringDataset(dataset_dir, "filenames/springtest.txt", opts.val_height, opts.val_width, mode='test')
    else:
        raise ValueError(f'Dataset not recognized: {opts.dataset}')

    data_loader_train = None
    data_loader_validation = None
    data_loader_test = None
    if dataset_train:
        data_loader_train = data.DataLoader(dataset_train, batch_size=opts.batch_size, shuffle=True,
                                            num_workers=opts.num_workers, pin_memory=False)
    if dataset_validation:
        data_loader_validation = data.DataLoader(dataset_validation, batch_size=opts.val_batch_size, shuffle=False,
                                             num_workers=opts.num_workers, pin_memory=False)
    if dataset_test:
        data_loader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=False,
                                       num_workers=0, pin_memory=False)

    return data_loader_train, data_loader_validation, data_loader_test
