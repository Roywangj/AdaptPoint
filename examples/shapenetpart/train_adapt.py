#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 20:59
# @Author  : wangjie

"""
Distributed training script for scene segmentation with S3DIS dataset
"""
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
from openpoints.models_adaptpoint import build_adaptpointmodels_from_cfg
from openpoints.function_adaptpoint import Form_dataset_shapenet
from openpoints.online_aug.pointwolf import PointWOLF_classversion
from openpoints.dataset.shapenetpart_c.shapenetpart_c import ShapeNetPartC, eval_corrupt_wrapper_shapenetc
from openpoints.utils import Summary
import h5py


def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def part_seg_refinement(pred, pos, cls, cls2parts, n=10):
    pred_np = pred.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):  # sample_idx
        parts = cls2parts[cls[shape_idx]]
        counter_part = Counter(pred_np[shape_idx])
        if len(counter_part) > 1:
            for i in counter_part:
                if counter_part[i] < n or i not in parts:
                    less_idx = np.where(pred_np[shape_idx] == i)[0]
                    less_pos = pos[shape_idx][less_idx]
                    knn_idx = knn_point(n + 1, torch.unsqueeze(less_pos, axis=0),
                                        torch.unsqueeze(pos[shape_idx], axis=0))[1]
                    neighbor = torch_grouping_operation(pred[shape_idx:shape_idx + 1].unsqueeze(1), knn_idx)[0][0]
                    counts = batched_bincount(neighbor, 1, cls2parts[-1][-1] + 1)
                    counts[:, i] = 0
                    pred[shape_idx][less_idx] = counts.max(dim=1)[1]
    return pred


def get_ins_mious(pred, target, cls, cls2parts,
                  multihead=False,
                  ):
    """Get the Shape IoU
    shape IoU: the mean part iou for each shape
    Args:
        pred (_type_): _description_
        target (_type_): _description_
        num_classes (_type_): _description_
    Returns:
        _type_: _description_
    """
    ins_mious = []
    for shape_idx in range(pred.shape[0]):  # sample_idx
        part_ious = []
        parts = cls2parts[cls[shape_idx]]
        if multihead:
            parts = np.arange(len(parts))

        for part in parts:
            pred_part = pred[shape_idx] == part
            target_part = target[shape_idx] == part
            I = torch.logical_and(pred_part, target_part).sum()
            U = torch.logical_or(pred_part, target_part).sum()
            if U == 0:
                iou = torch.tensor(100, device=pred.device, dtype=torch.float32)
            else:
                iou = I * 100 / float(U)
            part_ious.append(iou)
        ins_mious.append(torch.mean(torch.stack(part_ious)))
    return ins_mious

def copyfiles(cfg):
    import shutil
    #   copy pointcloud model
    path_copy = f'{cfg.run_dir}/copyfile'
    if not os.path.isdir(path_copy):
        os.makedirs(path_copy)
    shutil.copy(f'{os.path.realpath(__file__)}', path_copy)
    shutil.copytree('openpoints', f'{path_copy}/openpoints')
    pass

def get_gan_model(cfg):
    """
    return PointAug augmentor and discriminator
    and corresponding optimizer and scheduler
    """
    # Create model: G and D
    print("==> Creating model...")
    # generator
    generator = build_adaptpointmodels_from_cfg(cfg.adaptmodel_gan).cuda()
    # generator = PointWOLF_classversion4().to(device)
    print("==> Total parameters of Generator: {:.2f}M"\
          .format(sum(p.numel() for p in generator.parameters()) / 1000000.0))

    # discriminator
    discriminator = build_adaptpointmodels_from_cfg(cfg.adaptmodel_dis).cuda()
    print("==> Total parameters of Discriminater: {:.2f}M"\
          .format(sum(p.numel() for p in discriminator.parameters()) / 1000000.0))

    if cfg.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)


    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=cfg.adaptpoint_params.lr_generator, betas=(cfg.adaptpoint_params.b1, cfg.adaptpoint_params.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=cfg.adaptpoint_params.lr_discriminator, betas=(cfg.adaptpoint_params.b1, cfg.adaptpoint_params.b2))

    criterion_gan = torch.nn.BCELoss()
    dict = {
        'model_G': generator,
        'model_D': discriminator,
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'criterion_gan': criterion_gan
    }
    return dict

def train_gan(cfg, gan_dict, train_loader, summary, writer, epoch, model_pointcloud):
    generator = gan_dict['model_G']
    discriminator = gan_dict['model_D']
    optimizer_G = gan_dict['optimizer_G']
    optimizer_D = gan_dict['optimizer_D']
    criterion_gan = gan_dict['criterion_gan']
    generator.train()
    discriminator.train()
    model_pointcloud.eval()
    # prepare buffer list for update
    tmp_pos_buffer_list = []
    tmp_y_buffer_list = []
    tmp_heights_buffer_list = []
    tmp_cls_buffer_list = []
    pointwolf = PointWOLF_classversion(**cfg.pointwolf)
    for i, data in tqdm(enumerate(train_loader), total=train_loader.__len__()):
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        points = data['pos']
        points_clone = points.clone()
        input_pointcloud = points[:, :, :3].contiguous()


        # pointwolf_WOpara = PointWOLF_classversion().to(device)
        _, pointcloud_pointwolf = pointwolf(input_pointcloud)
        real_label = torch.full((input_pointcloud.size(0), 1), 0.9, requires_grad=True).cuda()
        fake_label = torch.full((input_pointcloud.size(0), 1), 0.1, requires_grad=True).cuda()

        #  Train Generator
        _, gen_imgs = generator(input_pointcloud)
        g_loss_raw = criterion_gan(discriminator(gen_imgs), real_label)

        points[:, :, :3] = gen_imgs

        data_fake = {
            'pos': points[:, :, :3].contiguous(),
            'y': data['y'],
            'heights': data['heights'],
            'cls': data['cls']
        }
        data_real = {
            'pos': points_clone[:, :, :3].contiguous(),
            'y': data['y'],
            'heights': data['heights'],
            'cls': data['cls']
        }


        # feedback_loss = get_feedback_loss_ver1(cfg=cfg, model_pointcloud=model_pointcloud, \
        #                                   data_real=data_real, data_fake=data_fake, \
        #                                   epoch=epoch, summary=summary, writer=writer)
        #

        # g_loss = g_loss_raw + 1 * cls_loss_fakedata
        # g_loss = g_loss_raw + 1 * cls_loss_fakedata + 0.1 * feedback_loss
        # g_loss = g_loss_raw + feedback_loss * 1
        g_loss = g_loss_raw

        # print(f"gard before backward: {generator.predict_prob_layer.embedding.net[0].weight.grad}")
        optimizer_G.zero_grad()
        g_loss.backward()
        # print(f"gard after backward: {generator.predict_prob_layer.extract_local_feat_masking[0].weight.grad[0][0][0]}")
        # print(f"weight: {generator.predict_prob_layer.extract_local_feat_masking[0].weight[0][0][0]}")
        # print(f"gard after backward: {generator.predict_prob_layer.extract_local_feat_masking.net[0].weight.grad[0][0][0]}")
        # print(f"weight: {generator.predict_prob_layer.extract_local_feat_masking.net[0].weight[0][0][0]}")
        optimizer_G.step()
        writer.add_scalar('train_G_iter/gen_loss_raw', g_loss_raw.item(), summary.train_iter_num)
        # writer.add_scalar('train_G_iter/gen_loss_feedback', feedback_loss.item(), summary.train_iter_num)


        # ---------------------
        #  Train Discriminator
        # ---------------------
        if i % 1 == 0:
            """
            判别器损失函数：一方面让真实图片通过判别器与valid越接近，
            另一方面让生成的图片通过判别器与fake越接近(正好与生成器相矛盾，这样才能提高判别能力)
            """
            # real_loss = criterion_gan(discriminator(pointcloud_pointwolf), real_label)
            real_loss = criterion_gan(discriminator(input_pointcloud.detach()), real_label)
            # fake_loss = criterion_gan(discriminator(gen_imgs), fake_label)
            fake_loss = criterion_gan(discriminator(gen_imgs.detach()), fake_label)
            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

        writer.add_scalar('train_G_iter/gen_loss', g_loss.item(), summary.train_iter_num)
        writer.add_scalar('train_G_iter/dis_loss', d_loss.item(), summary.train_iter_num)
        summary.summary_train_iter_num_update()



        #   save fake_data each N mini-batch
        if (i) % 10 == 0 and i < 110:
            path = f'{cfg.run_dir}/fakedata/epoch{epoch}'
            if not os.path.isdir(path):
                os.makedirs(path)
            f = h5py.File(f'{path}/minibatch{i}.h5', 'w')  # 创建一个h5文件，文件指针是f
            f['pointcloud'] = gen_imgs.detach().cpu().numpy()  # 将数据写入文件的主键data下面
            f['raw'] = input_pointcloud.detach().cpu().numpy()
            f['raw_pointwolf'] = pointcloud_pointwolf.detach().cpu().numpy()
            f['label'] = data['y'].detach().cpu().numpy()
            f.close()

        tmp_pos_buffer_list.append(gen_imgs.detach().cpu().numpy())
        tmp_y_buffer_list.append(data['y'].detach().cpu().numpy())
        tmp_cls_buffer_list.append(data['cls'].detach().cpu().numpy())
        tmp_heights_buffer_list.append(data['heights'].detach().cpu().numpy())

        # print(f'{i}-th, g_loss:{g_loss}, d_loss:{d_loss}')

    # save_ganmodel(generator=generator, discriminator=discriminator, path=args.checkpoint)
    # buffer loader will be used to save fake pose pair
    print('\nprepare buffer loader for train on fake pose')
    model_pointcloud.zero_grad()
    fake_dataset = Form_dataset_shapenet(tmp_pos_buffer_list, tmp_y_buffer_list, tmp_heights_buffer_list, tmp_cls_buffer_list)

    return fake_dataset


def main(gpu, cfg):
    # copyfiles(cfg)
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        # writer = SummaryWriter(log_dir=cfg.run_dir)
        # summary = Summary(cfg.run_dir)
        # writer = summary.create_summary()
    else:
        writer = None
    summary = Summary(cfg.run_dir)
    writer = summary.create_summary()
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.batch_size,
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.cls2parts = val_loader.dataset.cls2parts
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    if cfg.model.get('decoder_args', False):
        cfg.model.decoder_args.cls2partembed = val_loader.dataset.cls2partembed
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).cuda()
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(
            model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')


    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # transforms
    if 'vote' in cfg.datatransforms:
        voting_transform = build_transforms_from_cfg('vote', cfg.datatransforms)
    else:
        voting_transform = None

    model_module = model.module if hasattr(model, 'module') else model
    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            test_ins_miou, test_cls_miou, test_cls_mious = validate_fn(model, val_loader, cfg,
                                                                            num_votes=cfg.num_votes,
                                                                            data_transform=voting_transform
                                                                            )

            logging.info(f'\nresume val instance mIoU is {test_ins_miou}, val class mIoU is {test_cls_miou} \n ')
        else:
            if cfg.mode in ['val', 'test']:
                load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                test_ins_miou, test_cls_miou, test_cls_mious = validate_fn(model, val_loader, cfg,
                                                                            num_votes=cfg.num_votes,
                                                                            data_transform=voting_transform
                                                                            )
                return test_ins_miou
            elif cfg.mode == 'finetune':
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, pretrained_path=cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                logging.info(f'Load encoder only, finetuning from {cfg.pretrained_path}')
                load_checkpoint(model_module.encoder, pretrained_path=cfg.pretrained_path)
    else:
        logging.info('Training from scratch')

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader.dataset, 'num_per_class'):
            cfg.criterion_args.weight = None
            cfg.criterion_args.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
        else:
            logging.info('`num_per_class` attribute is not founded in dataset')
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    # ===> start training
    best_ins_miou, cls_miou_when_best, cls_mious_when_best = 0., 0., []
    optimizer.zero_grad()
    gan_model_dict = get_gan_model(cfg)
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        # some dataset sets the dataset length as a fixed steps.
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        cfg.epoch = epoch

        fake_dataset = train_gan(cfg, gan_model_dict, train_loader, summary, writer, epoch, model)
        fake_train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                                 cfg.dataset,
                                                 cfg.dataloader,
                                                 datatransforms_cfg=cfg.datatransforms,
                                                 split='train',
                                                 distributed=cfg.distributed,
                                                 dataset=fake_dataset,
                                                 )

        # train_loss = \
        #     train_one_epoch(model, train_loader, criterion,
        #                     optimizer, scheduler, epoch, cfg)

        train_loss = \
            train_one_epoch(model, fake_train_loader, criterion,
                            optimizer, scheduler, epoch, cfg)

        if (epoch+1) % 20 == 0:
            eval_corrupt_wrapper_shapenetc(model, validate_shapenetc, {'cfg': cfg}, cfg.run_dir, epoch)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_ins_miou, val_cls_miou, val_cls_mious = validate_fn(model, val_loader, cfg)
            if val_ins_miou > best_ins_miou:
                best_ins_miou = val_ins_miou
                cls_miou_when_best = val_cls_miou
                cls_mious_when_best = val_cls_mious
                best_epoch = epoch
                is_best = True
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_ins_miou {best_ins_miou:.2f} val_cls_miou {cls_miou_when_best:.2f}, '
                        f'\ncls_mious: {cls_mious_when_best}')

        lr = optimizer.param_groups[0]['lr']
        if writer is not None:
            writer.add_scalar('val_ins_miou', val_ins_miou, epoch)
            writer.add_scalar('val_class_miou', val_cls_miou, epoch)
            writer.add_scalar('best_val_instance_miou',
                              best_ins_miou, epoch)
            writer.add_scalar('val_class_miou_when_best', cls_miou_when_best, epoch)
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('lr', lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)

        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'ins_miou': best_ins_miou,
                                             'cls_miou': cls_miou_when_best},
                            is_best=is_best
                            )
    # if writer is not None:
    #     Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Best Epoch {best_epoch},'
                     f'Instance mIoU {best_ins_miou:.2f}, '
                     f'Class mIoU {cls_miou_when_best:.2f}, '
                     f'\n Class mIoUs {cls_mious_when_best}')

    if cfg.get('num_votes', 0) > 0:
        load_checkpoint(model, pretrained_path=os.path.join(
            cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
        set_random_seed(cfg.seed)
        test_ins_miou, test_cls_miou, test_cls_mious  = validate_fn(model, val_loader, cfg, num_votes=cfg.get('num_votes', 0),
                                 data_transform=voting_transform)
        with np.printoptions(precision=2, suppress=True):
            logging.info(f'---Voting---\nBest Epoch {best_epoch},'
                        f'Voting Instance mIoU {test_ins_miou:.2f}, '
                        f'Voting Class mIoU {test_cls_miou:.2f}, '
                        f'\n Voting Class mIoUs {test_cls_mious}')

        if writer is not None:
            writer.add_scalar('test_ins_miou_voting', test_ins_miou, epoch)
            writer.add_scalar('test_class_miou_voting', test_cls_miou, epoch)
    torch.cuda.synchronize()
    best_ckpt_path = os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth')
    last_ckpt_path = os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_latest.pth')
    testshapenetc(model=model, path=best_ckpt_path, cfg=cfg)
    testshapenetc(model=model, path=last_ckpt_path, cfg=cfg)
    if writer is not None:
        writer.close()
    dist.destroy_process_group()
    wandb.finish(exit_code=True)


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


@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=0, data_transform=None):
    model.eval()  # set model to eval mode
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list = []

    # label_size: b, means each sample has one corresponding class
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        cls = data['cls']
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        batch_size, num_point, _ = data['pos'].size()
        logits = 0
        for v in range(num_votes+1):
            set_random_seed(v)
            if v > 0:
                data['pos'] = data_transform(data['pos'])
            logits += model(data)
        logits /= (num_votes + 1)
        preds = logits.max(dim=1)[1]
        if cfg.get('refine', False):
            part_seg_refinement(preds, data['pos'], data['cls'], cfg.cls2parts, cfg.get('refine_n', 10))

        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            batch_ins_mious = get_ins_mious(preds, target, data['cls'], cfg.cls2parts)
            ins_miou_list += batch_ins_mious
        else:
            iou_array = []
            for ib in range(batch_size):
                sl = data['cls'][ib][0]
                iou = get_ins_mious(preds[ib:ib + 1], target[ib:ib + 1], sl.unsqueeze(0), cfg.cls2parts,
                                    multihead=True)
                iou_array.append(iou)
            ins_miou_list += iou_array

        # per category iou at each batch_size:
        for shape_idx in range(batch_size):  # sample_idx
            cur_gt_label = int(cls[shape_idx].cpu().numpy())
            # add the iou belongs to this cat
            cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]
            cls_nums[cur_gt_label] += 1

    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
    if cfg.distributed:
        dist.all_reduce(cls_mious), dist.all_reduce(cls_nums), dist.all_reduce(ins_mious_sum), dist.all_reduce(count)

    for cat_idx in range(cfg.shape_classes):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)
    with np.printoptions(precision=2, suppress=True):
        logging.info(f'Test Epoch [{cfg.epoch}/{cfg.epochs}],'
                        f'Instance mIoU {ins_miou:.2f}, '
                        f'Class mIoU {cls_miou:.2f}, '
                        f'\n Class mIoUs {cls_mious}')
    return ins_miou, cls_miou, cls_mious

@torch.no_grad()
def validate_shapenetc(split, model, cfg, num_votes=0, data_transform=None):
    model.eval()  # set model to eval mode
    dataset_transform = build_transforms_from_cfg('val', cfg.datatransforms_shapenet_c)
    testloader = torch.utils.data.DataLoader(ShapeNetPartC(split=split, transform=dataset_transform), num_workers=int(cfg.dataloader.num_workers), \
                            batch_size=cfg.get('val_batch_size', cfg.batch_size), shuffle=False, drop_last=False)
    pbar = tqdm(enumerate(testloader), total=testloader.__len__())
    cls_mious = torch.zeros(cfg.shape_classes, dtype=torch.float32).cuda(non_blocking=True)
    cls_nums = torch.zeros(cfg.shape_classes, dtype=torch.int32).cuda(non_blocking=True)
    ins_miou_list = []

    # label_size: b, means each sample has one corresponding class
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        target = data['y']
        cls = data['cls']
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        batch_size, num_point, _ = data['pos'].size()
        logits = 0
        for v in range(num_votes+1):
            set_random_seed(v)
            if v > 0:
                data['pos'] = data_transform(data['pos'])
            logits += model(data)
        logits /= (num_votes + 1)
        preds = logits.max(dim=1)[1]
        if cfg.get('refine', False):
            part_seg_refinement(preds, data['pos'], data['cls'], cfg.cls2parts, cfg.get('refine_n', 10))

        if cfg.criterion_args.NAME != 'MultiShapeCrossEntropy':
            batch_ins_mious = get_ins_mious(preds, target, data['cls'], cfg.cls2parts)
            ins_miou_list += batch_ins_mious
        else:
            iou_array = []
            for ib in range(batch_size):
                sl = data['cls'][ib][0]
                iou = get_ins_mious(preds[ib:ib + 1], target[ib:ib + 1], sl.unsqueeze(0), cfg.cls2parts,
                                    multihead=True)
                iou_array.append(iou)
            ins_miou_list += iou_array

        # per category iou at each batch_size:
        for shape_idx in range(batch_size):  # sample_idx
            cur_gt_label = int(cls[shape_idx].cpu().numpy())
            # add the iou belongs to this cat
            cls_mious[cur_gt_label] += batch_ins_mious[shape_idx]
            cls_nums[cur_gt_label] += 1

    ins_mious_sum, count = torch.sum(torch.stack(ins_miou_list)), torch.tensor(len(ins_miou_list)).cuda()
    if cfg.distributed:
        dist.all_reduce(cls_mious), dist.all_reduce(cls_nums), dist.all_reduce(ins_mious_sum), dist.all_reduce(count)

    for cat_idx in range(cfg.shape_classes):
        # indicating this cat is included during previous iou appending
        if cls_nums[cat_idx] > 0:
            cls_mious[cat_idx] = cls_mious[cat_idx] / cls_nums[cat_idx]

    ins_miou = ins_mious_sum/count
    cls_miou = torch.mean(cls_mious)
    # with np.printoptions(precision=2, suppress=True):
    #     logging.info(f'Test Epoch [{cfg.epoch}/{cfg.epochs}],'
    #                     f'Instance mIoU {ins_miou:.2f}, '
    #                     f'Class mIoU {cls_miou:.2f}, '
    #                     f'\n Class mIoUs {cls_mious}')
    return {
        'acc': 0,
        'class_mIOU': cls_miou.item(),
        'ins_mIOU': ins_miou.item()
    }
    # return ins_miou, cls_miou, cls_mious

def testshapenetc(model, path, cfg):
    ckpt = torch.load(f'{path}')
    model.load_state_dict(ckpt['model'])
    epoch  = ckpt['epoch']
    eval_corrupt_wrapper_shapenetc(model, validate_shapenetc, {'cfg': cfg},
                               cfg.run_dir, epoch)
