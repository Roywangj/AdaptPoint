#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 16:55
# @Author  : wangjie

import __init__
import os, argparse, yaml, numpy as np
from torch import multiprocessing as mp
from examples.classification.train import main as train
# from examples.classification.pretrain import main as pretrain
from openpoints.utils import EasyConfig, dist_utils, find_free_port, generate_exp_directory, resume_exp_directory, Wandb
from openpoints.utils import EasyConfig
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataloader_from_cfg,build_dataset_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.utils import registry
from openpoints.transforms import build_transforms_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
import torch
from openpoints.models_adaptpoint import build_adaptpointmodels_from_cfg

def test_build_dataset(args, transform):
    dataset_cfg = args.dataset
    split_cfg = {}
    split_cfg["transform"] = transform
    split_cfg["split"] = 'train'

    dataset = build_dataset_from_cfg(dataset_cfg.common, split_cfg)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)

    # cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    model = build_model_from_cfg(cfg.model).cuda()
    model = model.eval()
    print(model)

    adaptmodel_gan = build_adaptpointmodels_from_cfg(cfg.adaptmodel_gan)
    #
    # train_loader = build_dataloader_from_cfg(cfg.batch_size,
    #                                          cfg.dataset,
    #                                          cfg.dataloader,
    #                                          datatransforms_cfg=cfg.datatransforms,
    #                                          split='train',
    #                                          distributed=False,
    #                                          )
    #
    # data_transform = build_transforms_from_cfg('train', cfg.datatransforms)
    # dataset = test_build_dataset(args=cfg, transform=data_transform)
    # data = dataset.__getitem__(1)
    # print(data)
    # for key in data.keys():
    #     data[key] = data[key].unsqueeze(dim=0).cuda()
    # data['x'] = data['x'].transpose(1, 2).contiguous()
    #     # data[key] = torch.tensor(data[key]).unsqueeze(dim=0).cuda()
    #     # torch.tensor(points).unsqueeze(dim=0).to(device)
    # pred = model(data)
    #
    # criterion = build_criterion_from_cfg(cfg.criterion).cuda()
    # # optimizer & scheduler
    # optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    # scheduler = build_scheduler_from_cfg(cfg, optimizer)
    # # optimizer2 = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=2e-4)
    print('==> ending')