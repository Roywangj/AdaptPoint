#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 15:18
# @Author  : wangjie
import os
import glob
import h5py
import json
import pickle
import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from openpoints.models.layers import fps, furthest_point_sample
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def load_data_partseg(split, DATA_DIR):
    # SHAPENET_C_DIR = os.path.join(DATA_DIR, 'shapenet_c')
    SHAPENET_C_DIR = DATA_DIR

    all_data = []
    all_label = []
    all_seg = []

    file = os.path.join(SHAPENET_C_DIR, '%s.h5'%split)
    f = h5py.File(file, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    seg = f['pid'][:].astype('int64')  # part seg label
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_seg.append(seg)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg

class ShapeNetPartC(Dataset):
    classes = ['airplane', 'bag', 'cap', 'car', 'chair',
               'earphone', 'guitar', 'knife', 'lamp', 'laptop',
               'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]


    cls_parts = {'earphone': [16, 17, 18], 'motorbike': [30, 31, 32, 33, 34, 35], 'rocket': [41, 42, 43],
                   'car': [8, 9, 10, 11], 'laptop': [28, 29], 'cap': [6, 7], 'skateboard': [44, 45, 46], 'mug': [36, 37],
                   'guitar': [19, 20, 21], 'bag': [4, 5], 'lamp': [24, 25, 26, 27], 'table': [47, 48, 49],
                   'airplane': [0, 1, 2, 3], 'pistol': [38, 39, 40], 'chair': [12, 13, 14, 15], 'knife': [22, 23]}
    cls2parts = []
    cls2partembed = torch.zeros(16, 50)
    for i, cls in enumerate(classes):
        idx = cls_parts[cls]
        cls2parts.append(idx)
        cls2partembed[i].scatter_(0, torch.LongTensor(idx), 1)
    part2cls = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in cls_parts.keys():
        for label in cls_parts[cat]:
            part2cls[label] = cat

    def __init__(self,
                 data_root='data/ShapeNetPart_C/shapenet_c',
                 num_points=2048,
                 split='train',
                 class_choice=None,
                 shape_classes=16, transform=None, **kwargs):
        self.data, self.label, self.seg = load_data_partseg(split, data_root)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16,
                            19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = split
        self.class_choice = class_choice
        self.transform = transform

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0
            self.eye = np.eye(shape_classes)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]
        # if self.partition == 'trainval':
        #     pointcloud = translate_pointcloud(pointcloud)
        #     indices = list(range(pointcloud.shape[0]))
        #     np.random.shuffle(indices)
        #     pointcloud = pointcloud[indices]
        #     seg = seg[indices]

        # this is model-wise one-hot enocoding for 16 categories of shapes
        feat = np.transpose(self.eye[label, ].repeat(pointcloud.shape[0], 0))
        data = {'pos': pointcloud,
                # 'x': feat,
                'cls': label,
                'y': seg}
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return self.data.shape[0]

import pprint
def eval_corrupt_wrapper_shapenetc(model, fn_test_corrupt, args_test_corrupt, path, epoch):
    """
    The wrapper helps to repeat the original testing function on all corrupted test sets.
    It also helps to compute metrics.
    :param model: model
    :param fn_test_corrupt: original evaluation function, returns a dict of metrics, e.g., {'acc': 0.93}
    :param args_test_corrupt: a dict of arguments to fn_test_corrupt, e.g., {'test_loader': loader}
    :return:
    """
    file = open(os.path.join(path, 'outcorruption.txt'), "a")
    file.write(f"epoch: {epoch} \n")
    corruptions = [
        'clean',
        'scale',
        'jitter',
        'rotate',
        'dropout_global',
        'dropout_local',
        'add_global',
        'add_local',
    ]
    OA_clean = None
    # perf_all = {'Acc': [], 'InsIOU': []}
    perf_all = {'acc': [], 'class_mIOU': [], 'ins_mIOU': []}
    # perf_all = {'OA': [], 'CE': [], 'RCE': []}
    for corruption_type in corruptions:
        perf_corrupt = {'acc': [], 'class_mIOU': [], 'ins_mIOU': []}
        for level in range(5):
            if corruption_type == 'clean':
                split = "clean"
            else:
                split = corruption_type + '_' + str(level)
            test_perf = fn_test_corrupt(split=split, model=model, **args_test_corrupt)
            if not isinstance(test_perf, dict):
                test_perf = {'acc': test_perf}

            perf_corrupt['acc'].append(test_perf['acc'])
            perf_corrupt['class_mIOU'].append(test_perf['class_mIOU'])
            perf_corrupt['ins_mIOU'].append(test_perf['ins_mIOU'])

            test_perf['corruption'] = corruption_type
            if corruption_type != 'clean':
                test_perf['level'] = level

            pprint.pprint(test_perf, width=200)
            file.write(f"{test_perf} \n")
            if corruption_type == 'clean':
                break
        for k in perf_corrupt:
            perf_corrupt[k] = sum(perf_corrupt[k]) / len(perf_corrupt[k])
            perf_corrupt[k] = round(perf_corrupt[k], 3)

        if corruption_type != 'clean':
            for k in perf_all:
                perf_corrupt[k] = round(perf_corrupt[k], 3)
                perf_all[k].append(perf_corrupt[k])
        perf_corrupt['corruption'] = corruption_type
        perf_corrupt['level'] = 'Overall'
        pprint.pprint(perf_corrupt, width=200)
        file.write(f"{perf_corrupt} \n")
    for k in perf_all:
        perf_all[k] = sum(perf_all[k]) / len(perf_all[k])
        perf_all[k] = round(perf_all[k], 3)
    perf_all['macc'] = perf_all.pop('acc')
    perf_all['mclass_mIOU'] = perf_all.pop('class_mIOU')
    perf_all['mins_mIOU'] = perf_all.pop('ins_mIOU')
    pprint.pprint(perf_all, width=200)
    file.write(f"{perf_all} \n")
    file.close()


