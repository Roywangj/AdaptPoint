#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 18:43
# @Author  : wangjie
import argparse
import yaml
import os
import sys
import logging
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter
import torch.nn.functional as F
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix
from collections import defaultdict, Counter

torch.backends.cudnn.benchmark = False
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

from openpoints.models import build_model_from_cfg
from openpoints.models.layers import torch_grouping_operation, knn_point
from openpoints.loss import build_criterion_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.dataset import build_dataloader_from_cfg, get_class_weights, get_features_by_keys
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.models.layers import furthest_point_sample
from openpoints.dataset.shapenetpart_c.shapenetpart_c import ShapeNetPartC, eval_corrupt_wrapper_shapenetc
from openpoints.online_aug.pointwolf import PointWOLF_classversion
from openpoints.online_aug import rsmix_provider

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        num_iter += 1
        batch_size, num_point, _ = data['pos'].size()
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']  #   [B, 2048]/[B, N]
        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        logits = model(data)
        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            loss = criterion(logits, target)
        else:
            loss = criterion(logits, target, data['cls'])

        loss.backward()

        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        loss_meter.update(loss.item(), n=batch_size)
        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.avg:.3f} "
                                 )
    train_loss = loss_meter.avg
    return train_loss

def train_one_epoch_pointwolf(model, train_loader, criterion, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    pointwolf = PointWOLF_classversion(**cfg.pointwolf)
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        num_iter += 1
        batch_size, num_point, _ = data['pos'].size()
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']  #   [B, 2048]/[B, N]
        points = data['pos']
        _, points[:, :, :3] = pointwolf(points[:, :, :3])
        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        logits = model(data)
        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            loss = criterion(logits, target)
        else:
            loss = criterion(logits, target, data['cls'])

        loss.backward()

        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
            num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        loss_meter.update(loss.item(), n=batch_size)
        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.avg:.3f} "
                                 )
    train_loss = loss_meter.avg
    return train_loss