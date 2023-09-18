#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 16:35
# @Author  : wangjie

import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
from examples.classification.train_pointwolf_utils import train_one_epoch_pointwolf, train_one_epoch_rsmix
from openpoints.models_adaptpoint import build_adaptpointmodels_from_cfg
from openpoints.function_adaptpoint import Form_dataset_cls, get_feedback_loss_ver1
from openpoints.online_aug.pointwolf import PointWOLF_classversion
import h5py
from openpoints.utils import Summary
from openpoints.dataset.scanobjectnn_c.scanobjectnn_c import ScanObjectNNC, eval_corrupt_wrapper_scanobjectnnc
from openpoints.loss import build_criterion_from_cfg

def copyfiles(cfg):
    import shutil
    #   copy pointcloud model
    path_copy = f'{cfg.run_dir}/copyfile'
    if not os.path.isdir(path_copy):
        os.makedirs(path_copy)
    shutil.copy(f'{os.path.realpath(__file__)}', path_copy)
    shutil.copytree('openpoints', f'{path_copy}/openpoints')
    pass

def get_features_by_keys(input_features_dim, data):
    if input_features_dim == 3:
        features = data['pos']
    elif input_features_dim == 4:
        features = torch.cat(
            (data['pos'], data['heights']), dim=-1)
        raise NotImplementedError("error")
    return features.transpose(1, 2).contiguous()


def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
    accs_table = [f'{item:.2f}' for item in accs]
    header = ['method', 'OA', 'mAcc'] + \
        cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
        str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def print_cls_results(oa, macc, accs, epoch, cfg):
    s = f'\nClasses\tAcc\n'
    for name, acc_tmp in zip(cfg.classes, accs):
        s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
    s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
    logging.info(s)

def save_ganmodel(generator, discriminator, path):
    state = {
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
    }
    filepath = os.path.join(path, f"model_gan.pth")
    # filepath = os.path.join(path, f"gan_last_checkpoint.pth")
    torch.save(state, filepath)

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
    tmp_out_buffer_list = []
    tmp_points_buffer_list = []
    tmp_label_buffer_list = []
    pointwolf = PointWOLF_classversion(**cfg.pointwolf)
    for i, data in tqdm(enumerate(train_loader), total=train_loader.__len__()):
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        points = data['x']
        label = data['y']
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
            'y': label,
            'x': points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous(),
        }
        data_real = {
            'pos': points_clone[:, :, :3].contiguous(),
            'y': label,
            'x': points_clone[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous(),
        }

        feedback_loss_ratio = cfg.get('feedbackloss_ratio', 1)
        if feedback_loss_ratio > 0:
            feedback_loss = get_feedback_loss_ver1(cfg=cfg, model_pointcloud=model_pointcloud, \
                                              data_real=data_real, data_fake=data_fake, \
                                              epoch=epoch, summary=summary, writer=writer)
            g_loss = g_loss_raw + feedback_loss * feedback_loss_ratio
        else:
            g_loss = g_loss_raw

        # g_loss = g_loss_raw + feedback_loss * feedback_loss_ratio
        # g_loss = g_loss_raw

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
            real_loss = criterion_gan(discriminator(input_pointcloud), real_label)
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
            f['label'] = label.detach().cpu().numpy()
            f.close()

        tmp_out_buffer_list.append(gen_imgs.detach().cpu().numpy())
        tmp_label_buffer_list.append(label.detach().cpu().numpy())
        tmp_points_buffer_list.append(points.detach().cpu().numpy())

        # print(f'{i}-th, g_loss:{g_loss}, d_loss:{d_loss}')

    # save_ganmodel(generator=generator, discriminator=discriminator, path=args.checkpoint)
    # buffer loader will be used to save fake pose pair
    print('\nprepare buffer loader for train on fake pose')
    model_pointcloud.zero_grad()
    save_ganmodel(generator=generator, discriminator=discriminator, path=cfg.run_dir)
    fake_dataset = Form_dataset_cls(tmp_out_buffer_list, tmp_label_buffer_list, tmp_points_buffer_list)

    return fake_dataset




def main(gpu, cfg, profile=False):
    copyfiles(cfg)
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
    if cfg.rank == 0 :
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        # writer = SummaryWriter(log_dir=cfg.run_dir)
        summary = Summary(cfg.run_dir)
        writer = summary.create_summary()
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

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

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    num_points = val_loader.dataset.num_points if hasattr(
        val_loader.dataset, 'num_points') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}, "
                 f"number of points sampled from dataset: {num_points}, "
                 f"number of points as model input: {cfg.num_points}")
    cfg.classes = cfg.get('classes', None) or val_loader.dataset.classes if hasattr(
        val_loader.dataset, 'classes') else None or np.range(num_classes)
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
            print_cls_results(oa, macc, accs, cfg.start_epoch, cfg)
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val = load_checkpoint(
                    model, pretrained_path=cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, test_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
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

    # ===> start training
    val_macc, val_oa, val_accs, best_val, macc_when_best, best_epoch = 0., 0., [], 0., 0., 0
    model.zero_grad()
    gan_model_dict = get_gan_model(cfg)
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1

        fake_dataset = train_gan(cfg, gan_model_dict, train_loader, summary, writer, epoch, model)
        fake_train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                                 cfg.dataset,
                                                 cfg.dataloader,
                                                 datatransforms_cfg=cfg.datatransforms,
                                                 split='train',
                                                 distributed=cfg.distributed,
                                                 dataset=fake_dataset,
                                                 )


        # if cfg.pointwolf is not None:
        #     train_loss, train_macc, train_oa, _, _ = \
        #         train_one_epoch_pointwolf(model, train_loader,
        #                         optimizer, scheduler, epoch, cfg)
        # else:
            # train_loss, train_macc, train_oa, _, _ = \
            #     train_one_epoch(model, train_loader,
            #                     optimizer, scheduler, epoch, cfg)
        if cfg.get('rsmix_params', None) is not None:
            train_loss, train_macc, train_oa, _, _ = \
                train_one_epoch_rsmix(model, train_loader,
                                optimizer, scheduler, epoch, cfg)
        else:
            train_loss, train_macc, train_oa, _, _ = \
                train_one_epoch(model, fake_train_loader,
                            optimizer, scheduler, epoch, cfg)
        if (epoch+1) % 20 == 0:
            eval_corrupt_wrapper_scanobjectnnc(model, validate_scanobjectnnc, {'cfg': cfg}, cfg.run_dir, epoch)



        is_best = False
        if epoch % cfg.val_freq == 0:
            val_macc, val_oa, val_accs, val_cm = validate_fn(
                model, val_loader, cfg)
            is_best = val_oa > best_val
            if is_best:
                best_val = val_oa
                macc_when_best = val_macc
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')
                print_cls_results(val_oa, val_macc, val_accs, epoch, cfg)

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_oa {train_oa:.2f}, val_oa {val_oa:.2f}, best val oa {best_val:.2f}')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_oa', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('mAcc_when_best', macc_when_best, epoch)
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('epoch', epoch, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
    # test the last epoch
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, epoch)
        writer.add_scalar('test_macc', test_macc, epoch)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, best_epoch)
        writer.add_scalar('test_macc', test_macc, best_epoch)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)

    best_ckpt_path = os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth')
    last_ckpt_path = os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_latest.pth')
    testscanobjectnnc(model=model, path=best_ckpt_path, cfg=cfg)
    testscanobjectnnc(model=model, path=last_ckpt_path, cfg=cfg)

    if writer is not None:
        writer.close()
    if cfg.distributed:
        dist.destroy_process_group()

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:
        for key in data.keys():
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        points = data['x']
        target = data['y']
        """ bebug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0])
        """
        num_curr_pts = points.shape[1]
        if num_curr_pts > npoints:  # point resampling strategy
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if  points.size(1) < point_all:
                point_all = points.size(1)
            fps_idx = furthest_point_sample(
                points[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(
                point_all, npoints, False)]
            points = torch.gather(
                points, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, points.shape[-1]))

        data['pos'] = points[:, :, :3].contiguous()
        data['x'] = points[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()
        logits, loss = model.get_logits_loss(data, target) if not hasattr(model, 'module') else model.module.get_logits_loss(data, target)
        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    macc, overallacc, accs = cm.all_acc()
    return loss_meter.avg, macc, overallacc, accs, cm



@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())
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
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    return macc, overallacc, accs, cm

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
    return {'acc': (overallacc/100)}

def testscanobjectnnc(model, path, cfg):
    ckpt = torch.load(f'{path}')
    model.load_state_dict(ckpt['model'])
    epoch  = ckpt['epoch']
    eval_corrupt_wrapper_scanobjectnnc(model, validate_scanobjectnnc, {'cfg': cfg},
                               cfg.run_dir, epoch)