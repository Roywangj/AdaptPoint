#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/8 16:42
# @Author  : wangjie

'''
    pred probs for PointWOLF, no offset         ---wangjie
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pointnet2_ops import pointnet2_utils
from ..models.layers import furthest_point_sample, three_nn, three_interpolate
from ..models.layers.group import ball_query
from .build import ADAPTMODELS
# import sys
# print(sys.path)
# sys.path.append('..')
# print(sys.path)


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )


    def forward(self, x):
        return self.net(x)

class ConvBN1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=False):
        super(ConvBN1D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        return self.net(x)

@ADAPTMODELS.register_module()
class AdaptPoint_Augmentor(nn.Module):
    def __init__(self, w_num_anchor=4, w_sigma=0.5, w_R_range=10, w_S_range=3, w_T_range=0.25):
        super().__init__()
        self.num_anchor = w_num_anchor
        self.sigma = w_sigma
        self.R_range = (-abs(w_R_range), abs(w_R_range))
        self.S_range = (1., w_S_range)
        self.T_range = (-abs(w_T_range), abs(w_T_range))
        self.w_R_range = w_R_range
        self.w_S_range = w_S_range
        self.w_T_range = w_T_range

        self.predict_prob_layer = SAComponent()


    def forward(self, xyz):
        """
        Input:
            xyz: [B, N, 3]
        Output:
            xyz: [B, N, 3]
        """
        # print(f"gard: {self.predict_prob_layer.embedding.net[0].weight.grad[0][0][0]}")
        # print(f"weight: {self.predict_prob_layer.embedding.net[0].weight[0][0][0]}")


        #   masking precess()


        M = self.num_anchor
        B, N, _ = xyz.shape
        fps_idx = furthest_point_sample(xyz.contiguous(), self.num_anchor).long()  # [B, M]
        xyz_anchor = self.index_points(xyz, fps_idx)                            #   [B, M, 3]

        xyz_repeat = xyz.unsqueeze(dim=1).repeat(1, self.num_anchor, 1, 1)      #   [B, M, N, 3]

        # Move to canonical space
        xyz_normalize = xyz_repeat - xyz_anchor.unsqueeze(dim=-2)               #   [B, M, N, 3]

        #############
        probs, masking = self.predict_prob_layer(xyz, fps_idx)     # probs: [B, N, 3]   masking: [B, N, 2]
        # probs = self.predict_prob_layer(xyz, fps_idx)     # [B, M, 3]
        # probs, offset = self.predict_prob_layer(xyz, fps_idx)     # [B, M, 3]
        # print('probs:', probs[0, :, :])

        # Local transformation at anchor point
        xyz_transformed = self.local_transformaton(xyz_normalize, probs)  # (B,M,N,3)
        # xyz_transformed = self.local_transformaton(xyz_normalize)  # (B,M,N,3)

        # Move to origin space
        xyz_transformed = xyz_transformed + xyz_anchor.reshape(B, M, 1, 3)  # (B,M,N,3)
        xyz_new = self.kernel_regression(xyz, xyz_anchor, xyz_transformed)  #   [B, N, 3]
        xyz_new = self.normalize(xyz_new)

        xyz_new = xyz_new * masking[:,:,0].unsqueeze(dim=-1)

        # xyz_new = xyz_new + offset * 0.1

        # print(f'probs for R: {probs[:, :, :3]}')
        # print(f'probs for S: {probs[:, :, 3:6]}')
        # print(f'probs for T: {probs[:, :, 6:9]}')
        # print(f'offset: {offset*0.1}')
        return xyz, xyz_new



    def index_points(self, points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points


    def kernel_regression(self, pos, pos_anchor, pos_transformed):
        """
        input :
            pos([B,N,3])
            pos_anchor([B,M,3])
            pos_transformed([B,M,N,3])

        output :
            pos_new([B,N,3]) : Pointcloud after weighted local transformation
        """
        B, M, N, _ = pos_transformed.shape

        # Distance between anchor points & entire points
        sub = pos_anchor.unsqueeze(dim=-2).repeat(1, 1, N, 1) - \
              pos.unsqueeze(dim=1).repeat(1, M, 1, 1)  # (B, M, N, 3)

        project_axis = self.get_random_axis(B, 1).to(pos.device)   #   [B, 1, 3]

        projection = project_axis.unsqueeze(dim=-2) * torch.eye(3).to(pos.device)   #   [B, 1, 3, 3]

        # Project distance
        sub = sub @ projection  # (B, M, N, 3)
        sub = torch.sqrt(((sub) ** 2).sum(dim=-1))  # (B, M, N)

        # Kernel regression
        weight = torch.exp(-0.5 * (sub ** 2) / (self.sigma ** 2)).to(pos.device)  # (B, M, N)

        pos_new = (weight.unsqueeze(dim=-1).repeat(1, 1, 1, 3) * pos_transformed).sum(dim=1)  # (B, N, 3)
        pos_new = (pos_new / weight.sum(dim=1, keepdims=True).transpose(1,2).contiguous())  # normalize by weight   [B, N, 3]

        return pos_new

    def local_transformaton(self, pos_normalize, prob):
        """
        input :
            pos_normalize([B,M,N,3])

        output :
            pos_normalize([B,M,N,3]) : Pointclouds after local transformation centered at M anchor points.
        """
        B, M, N, _ = pos_normalize.shape
        a = torch.Tensor(B, M, 3).uniform_(0, 1)
        transformation_dropout = torch.bernoulli(a).to(pos_normalize.device)     #   [B, M, 3]
        transformation_axis = self.get_random_axis(B, M).to(pos_normalize.device)  # [B, M, 3]

        prob_R = prob[:,:,0:3]                        #   [B, M, 3]
        prob_R = torch.tanh(prob_R)                       #   [B, M, 3]
        # prob_R = F.tanh(prob_R)                       #   [B, M, 3]
        prob_R = prob_R * self.w_R_range              #   [B, M, 3]
        degree = torch.tensor(math.pi).to(pos_normalize.device) * prob_R.to(pos_normalize.device) / 180.0 \
                 * transformation_dropout[:, :, 0:1]    #   [B, M, 3], sampling from (-R_range, R_range)

        # degree = torch.tensor(math.pi).to(pos_normalize.device) * torch.FloatTensor(B, M, 3).uniform_(*self.R_range).to(pos_normalize.device) / 180.0 \
        #          * transformation_dropout[:, :, 0:1]    #   [B, M, 3], sampling from (-R_range, R_range)


        prob_S = prob[:, :, 3:6]                      #  [B, M, 3]
        prob_S = torch.sigmoid(prob_S)                    #  [B, M, 3]
        # prob_S = F.sigmoid(prob_S)                    #  [B, M, 3]
        prob_S = prob_S * (self.w_S_range-1) + 1      #  [B, M, 3]
        scale = prob_S.to(pos_normalize.device) * transformation_dropout[:, :, 1:2]  # [B, M, 3], sampling from (1, S_range)


        # scale = torch.FloatTensor(B, M, 3).uniform_(*self.S_range).to(pos_normalize.device) * transformation_dropout[:, :, 1:2]  # [B, M, 3], sampling from (1, S_range)
        scale = scale * transformation_axis
        scale = scale + 1 * (scale == 0)  # Scaling factor must be larger than 1


        prob_T = prob[:, :, 6:9]                    #   [B, M, 3]
        prob_T = torch.tanh(prob_T)                     #   [B, M, 3]
        # prob_T = F.tanh(prob_T)                     #   [B, M, 3]
        prob_T = prob_T * self.w_T_range            #   [B, M, 3]
        trl = prob_T.to(pos_normalize.device) * transformation_dropout[:, :, 2:3]    # [B, M, 3], sampling from (1, S_range)


        # trl = torch.FloatTensor(B, M, 3).uniform_(*self.T_range).to(pos_normalize.device) * transformation_dropout[:, :, 2:3]    # [B, M, 3], sampling from (1, S_range)
        trl *= transformation_axis

        # Scaling Matrix
        S = scale.unsqueeze(dim=-2) * torch.eye(3).to(pos_normalize.device)  # scailing factor to diagonal matrix (M,3) -> (M,3,3)

        # Rotation Matrix
        sin = torch.sin(degree).unsqueeze(dim=-1)
        cos = torch.cos(degree).unsqueeze(dim=-1)
        sx, sy, sz = sin[:, :, 0], sin[:, :, 1], sin[:, :, 2]
        cx, cy, cz = cos[:, :, 0], cos[:, :, 1], cos[:, :, 2]

        R = torch.cat([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
                      sz * cy, sz * sy * sx + cz * cy, sz * sy * cx - cz * sx,
                      -sy, cy * sx, cy * cx], dim=-1).reshape(B, M, 3, 3)
        pos_normalize = pos_normalize @ R @ S + trl.reshape(B, M, 1, 3)


        return pos_normalize

    def get_random_axis(self, batch, n_axis):
        """
        input :
            batch(int)
            n_axis(int)

        output :
            axis([batch, n_axis,3]) : projection axis
        """
        axis = torch.randint(1, 8, (batch, n_axis))  # 1(001):z, 2(010):y, 3(011):yz, 4(100):x, 5(101):xz, 6(110):xy, 7(111):xyz
        m = 3
        axis = (((axis[:, :, None] & (1 << torch.arange(m)))) > 0).int()
        return axis

    def normalize(self, pos):
        """
        input :
            pos([B, N, 3])

        output :
            pos([B, N, 3]) : normalized Pointcloud
        """
        B, N, C = pos.shape
        pos = pos - pos.mean(axis=-2, keepdims=True)    #   [B, N, 3]
        scale = (1 / torch.sqrt((pos ** 2).sum(dim=-1)).max(dim=-1)[0]) * 0.999999   #   [B, 1]
        pos = scale.reshape(B, 1, 1).repeat(1, N, C) * pos
        # scale = (1 / torch.sqrt((pos ** 2).sum(dim=-1)).max()) * 0.999999   #   [B, 1]
        # pos = scale * pos
        return pos


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=False, activation='relu'):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        # self.extraction = PosExtraction(out_channel, blocks, groups=groups,
        #                                 res_expansion=res_expansion, bias=bias, activation=activation)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D'', N]
        """

        # dists, idx = pointnet2_utils.three_nn(xyz1.contiguous(), xyz2.contiguous())   #   dists: [B, N, 3], idx: [B, N, 3]
        dists, idx = three_nn(xyz1.contiguous(), xyz2.contiguous())   #   dists: [B, N, 3], idx: [B, N, 3]
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        # interpolated_points = pointnet2_utils.three_interpolate(points2.contiguous(), idx.int(), weight)    #   [B, D'', N]
        interpolated_points = three_interpolate(points2.contiguous(), idx.int(), weight)    #   [B, D'', N]

        # return interpolated_points
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        new_points = self.fuse(new_points)
        # new_points = self.extraction(new_points)
        return new_points

#   PointsetGrouper_paradigm + norm
class PointsetGrouper(nn.Module):
    def __init__(self, channel, reduce, kneighbors, radi, normalize="anchor", **kwargs):
        """
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param radi: radius of ball_query
        :param kwargs: others
        norm_way: batchnorm/pointmlpnorm/pointsetnorm
        """
        super(PointsetGrouper, self).__init__()
        self.reduce = reduce
        self.kneighbors = kneighbors
        self.radi = radi

        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        '''
        input:
            xyz:[B, N, 3]
            points:[B, N, C]
        output:
            new_xyz:[B, np, 3]
            new_point:[B, C, np]

        '''
        #   xyz: [B, N, 3]      points: [B, N, C]
        B, N, C = points.shape
        xyz = xyz.contiguous()  # xyz [B, N, 3]

        fps_idx = furthest_point_sample(xyz.contiguous(), xyz.shape[1]//self.reduce).long()          #   [B, npoint]
        new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))                         #   [B, npoint, 3]
        new_points = torch.gather(points, 1, fps_idx.unsqueeze(-1).expand(-1, -1, points.shape[-1]))    #   [B, npoint, C]


        # ballquery_idx = pointnet2_utils.ball_query(self.radi, self.kneighbors, xyz, new_xyz).long()     #   [B, npoint, k]
        ballquery_idx = ball_query(self.radi, self.kneighbors, xyz, new_xyz).long()     #   [B, npoint, k]
        grouped_xyz = index_points(xyz, ballquery_idx)                                                  #   [B, npoint, k, 3]
        grouped_points = index_points(points, ballquery_idx)                                            #   [B, npoint, k, C]


        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, C]
            # std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            # grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = (grouped_points-mean)                                                      #   [B, npoint, k, C]
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta                        #   [B, npoint, k, C]

        new_points = torch.max(grouped_points, dim=2)[0].permute(0, 2, 1)                               #   [B, C, npoint]
        return new_xyz, new_points


class Anchor_selfattention(nn.Module):
    def __init__(self, dim, head_num):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = int(self.dim // self.head_num)
        self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)

        self.pos_embedding = nn.Sequential(nn.Conv1d(3, self.dim, 1),
                                           nn.BatchNorm1d(self.dim)
                                           )
        self.res = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1),
                                 nn.BatchNorm1d(self.dim))

    def forward(self, x, xyz=None):
        '''
        Input:
            x: [B, M, C]
            xyz: [B, M, 3]
        Output:
            v: [B, M, C]
        '''
        B, M, C = x.shape

        gravity_center = torch.mean(xyz, dim=1, keepdim=True)       #   [B, 1, 3]
        relative_xyz = xyz - gravity_center                         #   [B, M, 3]
        rxyz_embedding = self.pos_embedding(relative_xyz.permute(0, 2, 1)).permute(0, 2, 1)     #   [B, M, C]

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)                   #   q,k,v:  [B, M, C]
        q = q + rxyz_embedding
        k = k + rxyz_embedding
        v = v + rxyz_embedding
        q = q.reshape(B, M, self.head_num, self.head_dim)           #   [B, M, head_num, C']
        q = q.permute(0, 2, 1, 3)                                   #   [B, head_num, M, C']
        k = k.reshape(B, M, self.head_num, self.head_dim)           #   [B, M, head_num, C']
        k = k.permute(0, 2, 1, 3)                                   #   [B, head_num, M, C']
        v = v.reshape(B, M, self.head_num, self.head_dim)           #   [B, M, head_num, C']
        v = v.permute(0, 2, 1, 3)                                   #   [B, head_num, M, C']
        attn = q @ k.transpose(-2, -1)                              #   [B, head_num, M, M]
        attn /= self.head_dim ** 0.5                                #   [B, head_num, M, M]
        attn = attn.softmax(dim=-1)                                 #   [B, head_num, M, M]
        v = attn @ v                                                #   [B, head_num, M, C']
        v = v.permute(0, 2, 1, 3)                                   #   [B, M, head_num, C']
        v = v.reshape(B, M, -1)                                     #   [B, M, C]
        v = self.res(v.permute(0, 2, 1)).permute(0, 2, 1)           #   [B, M, C]

        return v



class Self_attention(nn.Module):
    def __init__(self, dim, head_num):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = int(self.dim // self.head_num)
        self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)

        self.pos_embedding = nn.Sequential(nn.Conv1d(3, self.dim, 1),
                                           nn.BatchNorm1d(self.dim)
                                           )
        self.res = nn.Sequential(nn.Conv1d(self.dim, self.dim, 1),
                                 nn.BatchNorm1d(self.dim))

    def forward(self, x, xyz=None):
        '''
        Input:
            x: [B, M, C]
            xyz: [B, M, 3]
        Output:
            v: [B, M, C]
        '''
        B, M, C = x.shape

        gravity_center = torch.mean(xyz, dim=1, keepdim=True)       #   [B, 1, 3]
        relative_xyz = xyz - gravity_center                         #   [B, M, 3]
        rxyz_embedding = self.pos_embedding(relative_xyz.permute(0, 2, 1)).permute(0, 2, 1)     #   [B, M, C]

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)                   #   q,k,v:  [B, M, C]
        q = q + rxyz_embedding
        k = k + rxyz_embedding
        v = v + rxyz_embedding
        q = q.reshape(B, M, self.head_num, self.head_dim)           #   [B, M, head_num, C']
        q = q.permute(0, 2, 1, 3)                                   #   [B, head_num, M, C']
        k = k.reshape(B, M, self.head_num, self.head_dim)           #   [B, M, head_num, C']
        k = k.permute(0, 2, 1, 3)                                   #   [B, head_num, M, C']
        v = v.reshape(B, M, self.head_num, self.head_dim)           #   [B, M, head_num, C']
        v = v.permute(0, 2, 1, 3)                                   #   [B, head_num, M, C']
        attn = q @ k.transpose(-2, -1)                              #   [B, head_num, M, M]
        attn /= self.head_dim ** 0.5                                #   [B, head_num, M, M]
        attn = attn.softmax(dim=-1)                                 #   [B, head_num, M, M]
        v = attn @ v                                                #   [B, head_num, M, C']
        v = v.permute(0, 2, 1, 3)                                   #   [B, M, head_num, C']
        v = v.reshape(B, M, -1)                                     #   [B, M, C]
        v = self.res(v.permute(0, 2, 1)).permute(0, 2, 1)           #   [B, M, C]

        return v



class Producefactor(nn.Module):
    def __init__(self, kneighbors, out_channels):
        super().__init__()
        self.keighbors = kneighbors
        self.out_channels = out_channels

        self.global_layer = nn.Sequential(
                                            nn.Conv1d(3, self.out_channels, 1, bias=False),
                                            nn.BatchNorm1d(out_channels)
                                          )

        self.prob_head = nn.Sequential(
                                            nn.Conv1d(self.out_channels*2, 3*3, 1, bias=False),
                                            nn.BatchNorm1d(3*3)
        )

        self.anchor_selfattention = Anchor_selfattention(dim=self.out_channels, head_num=4)



    def forward(self, a_points, sa_x, sa_xyz, xyz_raw):
        """
        Input:
            a_points:       [B, 4, 3]       anchor_points
            sa_x:           [B, np, C]      xyz after SA Modules
            sa_xyz:         [B, np, 3]      feat after SA Modules
            xyz_raw:          [B, N, 3]       feat after embedding
        Output:
            prob:           [B, 4, 3*3]       probilities of Rotation, Scaling, Translation(each has 3 axis xyz)
        """
        _,num_anchor,_ = a_points.shape
        idx_knn = knn_point(self.keighbors, sa_xyz, a_points)              #   [B, 4, k]
        local_feat = index_points(sa_x, idx_knn)                            #   [B, 4, k, C]
        local_feat = torch.max(local_feat, dim=2)[0]                        #   [B, 4, C]



        #   self attention
        local_feat_res = self.anchor_selfattention(x=local_feat, xyz=a_points)      #   [B, 4, C]
        local_feat = local_feat + local_feat_res


        #   extract global feat
        # global_feat = self.global_layer(xyz_raw.permute(0, 2, 1)).permute(0, 2, 1)               #   [B, N, C]
        global_feat = self.global_layer(a_points.permute(0, 2, 1)).permute(0, 2, 1)               #   [B, 4, C]
        global_feat = torch.max(global_feat, dim=1, keepdim=True)[0]          #   [B, 1, C]


        feat = torch.cat([local_feat, global_feat.repeat(1, num_anchor, 1)], dim=-1)     #   [B, 4, 2C]
        prob = self.prob_head(feat.permute(0, 2, 1)).permute(0, 2, 1)           #   [B, 4, 3*3]

        return prob


class SAComponent(nn.Module):
    def __init__(self, in_channel=3, embed_dim=64, res_expansion=1.0,
                 activation="relu", bias=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], radii= [0.1, 0.2, 0.4, 0.8],
                 k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], **kwargs):
        super().__init__()
        self.stages = len(dim_expansion)

        self.embedding = ConvBNReLU1D(in_channel, embed_dim, bias=bias, activation=activation)



        self.extract_feat_list = nn.ModuleList()
        self.pointset_grouper_list = nn.ModuleList()
        last_channel = embed_dim
        channels=[embed_dim]
        for i in range(len(dim_expansion)):
            out_channel = last_channel * dim_expansion[i]

            pre_block_module = ConvBNReLU1D(in_channels=last_channel, out_channels=out_channel,
                                                kernel_size=1, bias=bias, activation=activation)
            self.extract_feat_list.append(pre_block_module)

            SAModule = PointsetGrouper(channel=out_channel, reduce=reducers[i], kneighbors=k_neighbors[i], \
                                       radi=radii[i], normalize=normalize,)
            self.pointset_grouper_list.append(SAModule)
            last_channel = out_channel
            channels.append(out_channel)

        self.head = Producefactor(kneighbors=24, out_channels=last_channel)


        #   FP layer
        self.decode_list = nn.ModuleList()
        for i in range(self.stages):
            self.decode_list.append(
                PointNetFeaturePropagation(channels[-(i+1)]+channels[-(i+2)], channels[-(i+2)],
                                           blocks=1, groups=1, res_expansion=res_expansion,
                                           bias=bias, activation=activation)
            )

        # self.extract_local_feat_offset = nn.Sequential(
        #             nn.Conv1d(in_channels=embed_dim, out_channels=3, kernel_size=1, bias=False),
        #             nn.BatchNorm1d(3),
        # )
        # self.extract_global_feat_offset = nn.Sequential(
        #             nn.Conv1d(in_channels=last_channel, out_channels=3, kernel_size=1, bias=False),
        #             nn.BatchNorm1d(3),
        # )
        # self.fuse_offset = nn.Sequential(
        #             nn.Conv1d(in_channels=6, out_channels=3, kernel_size=1, bias=False),
        #             nn.BatchNorm1d(3),
        #             nn.Tanh()
        # )

        #
        self.localfeat_mask_selfattention = Anchor_selfattention(dim=embed_dim, head_num=4)
        self.extract_local_feat_masking = nn.Sequential(
                    nn.Conv1d(in_channels=embed_dim, out_channels=3, kernel_size=1, bias=False),
                    nn.BatchNorm1d(3),
        )
        self.extract_global_feat_masking = nn.Sequential(
                    nn.Conv1d(in_channels=last_channel, out_channels=3, kernel_size=1, bias=False),
                    nn.BatchNorm1d(3),
        )
        self.fuse_masking = nn.Sequential(
                    nn.Conv1d(in_channels=6, out_channels=2, kernel_size=1, bias=False),
                    nn.BatchNorm1d(2),
        )


    def forward(self, x, a_index=None):
        """
        Input:
            x: [B, N, 3]
            a_index: [B, M]     index of anchor points
        Output:
            prob: [B, 4, 3]  used to control the factors of PointWOLF
            offset: [B, N, 3] offset of points
        """
        _, N, _ = x.shape
        a_points = index_points(x, a_index)                         #   [B, M, 3]
        xyz = x                                                     #   [B, N, 3]
        x = x.permute(0, 2, 1).contiguous()                         #   [B, 3, N]
        x = self.embedding(x)                                       #   [B, C, N]
        xyz_raw = xyz

        xyz_list = [xyz]
        x_list = [x]
        for i in range(self.stages):
            #   Give feat[B, C, N], return new_feat[B, C_out, N]
            x = self.extract_feat_list[i](x)

            # Give xyz[B, N, 3] and feat[B, N, C], return new_xyz[B, np, 3] and new_fea[B, C, np]
            xyz, x = self.pointset_grouper_list[i](xyz, x.permute(0, 2, 1))
            xyz_list.append(xyz)
            x_list.append(x)


        for i in range(self.stages):        #   updating all features(x_list)
            x_list[-(i+2)] = self.decode_list[i](xyz1=xyz_list[-(i+2)], xyz2=xyz_list[-(i+1)], \
                                                 points1=x_list[-(i+2)], points2=x_list[-(i+1)])

        prob = self.head(a_points=a_points, sa_x=x.permute(0, 2, 1), sa_xyz=xyz, xyz_raw=xyz_raw)



        # #   offset
        # #   local tract
        # offset_local = self.extract_local_feat_offset(x_list[0])      #   [B, 3, N]
        # #   global tract
        # offset_global = self.extract_global_feat_offset(x_list[-1])   #   [B, 3, N']
        # offset_global = torch.max(offset_global, dim=2, keepdim=True)[0]                  #   [B, 3, 1]
        # offset = torch.cat([offset_local, offset_global.repeat(1,1,N)], dim=1)          #   [B, 6, N]
        # offset = self.fuse_offset(offset).permute(0, 2, 1)                              #   [B, N, 3]

        #   masking matrix
        #   local tract
        mask_localfeat = self.localfeat_mask_selfattention(x=x_list[0].permute(0, 2, 1), xyz=xyz_list[0])   #   [B, N, C]
        mask_localfeat = mask_localfeat + x_list[0].permute(0, 2, 1)                            #   [B, N, C]
        masking_local = self.extract_local_feat_masking(mask_localfeat.permute(0, 2, 1))      #   [B, 3, N]
        #   global tract
        masking_global = self.extract_global_feat_masking(x_list[-1])   #   [B, 3, N']
        masking_global = torch.max(masking_global, dim=2, keepdim=True)[0]                  #   [B, 3, 1]
        masking = torch.cat([masking_local, masking_global.repeat(1,1,N)], dim=1)          #   [B, 6, N]
        masking = self.fuse_masking(masking).permute(0, 2, 1)                              #   [B, N, 2]
        masking = F.gumbel_softmax(masking, tau=0.1, hard=True, eps=1e-10, dim=- 1)         #   [B, N, 2]



        return prob, masking
        # return prob
        # return prob, offset







if __name__ == '__main__':
    data = torch.rand(2, 1024, 3).cuda()
    print("===> testing pointMLP ...")
    model = PointWOLF_classversion4_15().cuda()
    model = model.eval()
    # print(model)
    # model(data)
    raw_data,newdata = model(data)
    # newdata = model(data)
    print(newdata.shape)

    from thop import profile
    flops, params = profile(model, (data,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')


