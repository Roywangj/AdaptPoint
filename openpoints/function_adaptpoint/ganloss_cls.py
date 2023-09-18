#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 21:02
# @Author  : wangjie
import torch
import torch.nn.functional as F
from ..loss import build_criterion_from_cfg


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss_raw = -(one_hot * log_prb).sum(dim=1)
        loss = loss_raw.mean()

    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss, loss_raw

# def get_feedback_loss_ver2_1(cfg, model_pointcloud, data_real, data_fake, summary, writer):
def get_feedback_loss_ver1(cfg, model_pointcloud, data_real, data_fake, epoch, summary, writer):
    '''
    To generate harder case
    '''
    def update_hardratio(start, end, current_epoch, total_epoch):
        return start + (end - start) * current_epoch / total_epoch

    def fix_hard_ratio_loss(expected_hard_ratio, harder, easier):  # similar to MSE
        fix_loss = torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))
        return fix_loss.mean()
        # return torch.abs(1 - torch.exp(harder - expected_hard_ratio * easier))

    #   get loss on real/fake data
    model_pointcloud.eval()
    pred_fake = model_pointcloud.forward(data_fake)                     #   [B, 40]
    pred_real = model_pointcloud.forward(data_real)                     #   [B, 40]
    label = data_real['y']
    criterion = build_criterion_from_cfg(cfg.criterion_args)
    # loss_fake, loss_raw_fake = criterion(pred_fake, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    # loss_real, loss_raw_real = criterion(pred_real, label.long())       #   loss_real: [1]   loss_raw_real: [B]
    loss_fake = criterion(pred_fake, label.long())       #   loss_fake: [1]   loss_raw_fake: [B]
    loss_real = criterion(pred_real, label.long())       #   loss_real: [1]   loss_raw_real: [B]


    #   updata hardratio
    hardratio = update_hardratio(cfg.adaptpoint_params.hardratio_s, cfg.adaptpoint_params.hardratio, epoch, cfg.epochs)

    # feedback_loss = fix_hard_ratio_loss(hardratio, loss_raw_fake, loss_raw_real)
    feedback_loss = fix_hard_ratio_loss(hardratio, loss_fake, loss_real)

    writer.add_scalar('train_G_iter/loss_fakedata', loss_fake.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/loss_realdata', loss_real.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/loss_ratio', loss_fake.item()/loss_real.item(), summary.train_iter_num)
    writer.add_scalar('train_G_iter/hardratio', hardratio, summary.train_iter_num)
    return feedback_loss
