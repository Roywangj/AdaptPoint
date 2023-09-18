#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 13:56
# @Author  : wangjie

import __init__
import os, argparse, yaml, numpy as np
from torch import multiprocessing as mp
from examples.classification.train import main as train
from openpoints.utils import EasyConfig, dist_utils, find_free_port, generate_exp_directory, resume_exp_directory, Wandb
from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg,build_dataset_from_cfg
from openpoints.utils import registry
from openpoints.transforms import build_transforms_from_cfg
import torch
from easydict import EasyDict as edict
import datetime
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from tqdm import tqdm
from openpoints.dataset.scanobjectnn_c.scanobjectnn_c import ScanObjectNNC, eval_corrupt_wrapper
import sklearn.metrics as metrics
import h5py


def build_dataset_own(cfg=None, mode='val'):
    #   build datatransforms
    data_transform = build_transforms_from_cfg(mode, cfg.datatransforms)
    split_cfg = cfg.dataset.get(mode, edict())
    split_cfg.transform = data_transform
    dataset = build_dataset_from_cfg(cfg.dataset.common, split_cfg)
    return dataset



def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)



@torch.no_grad()
def validate_scanobjectnnc(split, model, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    data_transform = build_transforms_from_cfg('val', cfg.datatransforms_scanobjectnn_c)
    testloader = torch.utils.data.DataLoader(ScanObjectNNC(split=split, transform=data_transform), num_workers=int(cfg.dataloader.num_workers), \
                            batch_size=cfg.get('val_batch_size', cfg.batch_size), shuffle=False, drop_last=False)
    pbar = tqdm(enumerate(testloader), total=testloader.__len__())
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        points = data['x']
        points = points[:, :npoints]
        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits = model(data)
        cm.update(logits.argmax(dim=1), target)

    tp, count = cm.tp, cm.count
    # if cfg.distributed:
    #     dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    return {'acc': overallacc}
    # return macc, overallacc, accs, cm

if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true',
                        default=False, help='set to True to profile speed')
    parser.add_argument('--msg_ckpt', default='ngpus1-seed6687-20230208-113827-C5JvDVLy53nEazUYXPtNyj', type=str, help='message after checkpoint')
    parser.add_argument('--testingmode', default='val', type=str, help='mode')

    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    cfg.msg_ckpt = args.msg_ckpt
    cfg.testingmode = args.testingmode
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    #   loading model
    model = build_model_from_cfg(cfg.model).cuda()
    # print(model)

    #   loading ckpt dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]
    cfg.exp_name = args.cfg.split('.')[-2].split('/')[-1]
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.exp_name,  # cfg file name
    ]
    if isinstance(tags, list):
        exp_name = '-'.join(tags)
    cfg.run_name = '-'.join([exp_name, cfg.msg_ckpt])
    cfg.run_dir = os.path.join(cfg.root_dir, cfg.run_name)
    cfg.ckpt_dir = os.path.join(cfg.run_dir, 'checkpoint')
    pretrained_path = os.path.join(cfg.ckpt_dir, os.path.join(cfg.run_name + '_ckpt_best.pth'))
    if not os.path.exists(pretrained_path):
        raise NotImplementedError('no checkpoint file from path %s...' % pretrained_path)






    #   loading state dict
    state_dict = torch.load(pretrained_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])

    #   loading dataset
    # dataset = build_dataset_own(cfg, cfg.testingmode)
    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=cfg.get('val_batch_size', cfg.batch_size),
    #                                          num_workers=int(cfg.dataloader.num_workers),
    #                                          worker_init_fn=worker_init_fn,
    #                                          drop_last=cfg.testingmode == 'train',
    #                                          # collate_fn=collate_fn,
    #                                          pin_memory=True
    #                                          )
    # test_macc, test_oa, test_accs, test_cm = validate(model, dataloader, cfg)
    # print(f'test oa: {test_oa}')

    print(' ==> starting testing on scanobjectnnc ...')
    eval_corrupt_wrapper(model, validate_scanobjectnnc, {'cfg': cfg}, cfg.run_dir)
    print(' ==> endinging testing on scanobjectnnc ...')



    # data_clean = ModelNetC(split='clean')
    # data_scale0 = ModelNetC(split='scale_0')
    # data = data_clean.__getitem__(0)
    # print(f"data_clean size: {data_clean.__len__()}")
    # print(f"data_scale0 size: {data_scale0.__len__()}")
    # print(f"data.shape: {data['x'].shape}")
    # print(f"label.shape: {data['y'].shape}")
    # print(f"label: {data['y']}")


