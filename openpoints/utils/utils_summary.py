#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 23:34
# @Author  : wangjie
from torch.utils.tensorboard import SummaryWriter
import os

class Summary(object):
    def __init__(self, directory):
        self.directory = directory
        self.epoch = 0
        self.writer = None
        self.phase = 0
        self.train_iter_num = 0
        self.train_realpose_iter_num = 0
        self.train_fakepose_iter_num = 0
        self.test_iter_num = 0
        self.test_MPI3D_iter_num = 0

    def create_summary(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return self.writer

    def summary_train_iter_num_update(self):
        self.train_iter_num = self.train_iter_num + 1

    def summary_train_realpose_iter_num_update(self):
        self.train_realpose_iter_num = self.train_realpose_iter_num + 1

    def summary_train_fakepose_iter_num_update(self):
        self.train_fakepose_iter_num = self.train_fakepose_iter_num + 1

    def summary_test_iter_num_update(self):
        self.test_iter_num = self.test_iter_num + 1

    def summary_test_MPI3D_iter_num_update(self):
        self.test_MPI3D_iter_num = self.test_MPI3D_iter_num + 1

    def summary_epoch_update(self):
        self.epoch = self.epoch + 1

    def summary_phase_update(self):
        self.phase = self.phase + 1
