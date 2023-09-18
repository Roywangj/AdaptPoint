#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/9 21:23
# @Author  : wangjie
import os
import glob
import h5py
import numpy as np
import open3d as o3d

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../fake_data')

red_rgb = np.array([255 / 255., 107 / 255., 107 / 255.])
green_rgb = np.array([107 / 255., 203 / 255., 119 / 255.])
blue_rgb = np.array([77 / 255, 150 / 255, 255 / 255])
purple = np.array([138 / 255, 163 / 255, 255 / 255])
red2green = red_rgb - green_rgb
green2blue = green_rgb - blue_rgb
red2blue = red_rgb - blue_rgb

classes = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]

def o3dvis(points_1, points_2, points_3=None, id=None):
    '''
        points_1: coors of feat,[npoint, 3]
        points_2: coors of feat,[npoint, 3]
    '''
    width = 1080
    height =1080


    front_size = [-0.078881283843093356, 0.98265290049739873, -0.16784224796908237]
    lookat_size = [0.057118464194894254, -0.010695673712330742, 0.047245129152854129]
    up_size = [0.018854223129214469, -0.16686616421723113, -0.98579927039414161]
    zoom = 0.98
    front = np.array(front_size)
    lookat = np.array(lookat_size)
    up = np.array(up_size)

    chessboard_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=10, origin=[0.1, 0.1, 0.1])




    color_1 = np.zeros_like(points_1)
    color_1[:] = red_rgb
    P_1 = o3d.geometry.PointCloud()
    P_1.points = o3d.utility.Vector3dVector(points_1)
    P_1.colors = o3d.utility.Vector3dVector(color_1)

    points_2 = points_2 - [1, 0, 0]
    color_2 = np.zeros_like(points_2)
    color_2[:] = blue_rgb
    # color_new[:] = green_rgb
    P_2 = o3d.geometry.PointCloud()
    P_2.points = o3d.utility.Vector3dVector(points_2)
    P_2.colors = o3d.utility.Vector3dVector(color_2)

    if points_3 is not None:
        points_3 = points_3 - [2, 0, 0]
        color_3 = np.zeros_like(points_3)
        color_3[:] = green_rgb
        # color_new[:] = green_rgb
        P_3 = o3d.geometry.PointCloud()
        P_3.points = o3d.utility.Vector3dVector(points_3)
        P_3.colors = o3d.utility.Vector3dVector(color_3)

    window_name = f"{id}"
    o3d.visualization.draw_geometries([P_1, P_2, P_3], window_name=window_name, width=width, height=height,
                                      zoom=zoom, front=front, lookat=lookat, up=up)








def load_h5_new(h5_name):
    f = h5py.File(h5_name, 'r')
    data_raw = f['raw'][:].astype('float32')
    data_raw_wpointwolf = f['raw_pointwolf'][:].astype('float32')
    data_fake = f['pointcloud'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    return data_raw, data_raw_wpointwolf, data_fake, label


if __name__ == "__main__":
    epoch = 7
    minibatch = 0

    path = f"../fake_data/epoch{epoch}/minibatch{minibatch}.h5"
    data_raw, data_raw_wpointwolf, data_fake, label= load_h5_new(h5_name=path)


    for i in range(data_raw.shape[0]):
        raw = data_raw[i]
        fake = data_fake[i]
        label_ = label[i]
        raw_wpointwolf = data_raw_wpointwolf[i]
        o3dvis(points_1=raw, points_2=raw_wpointwolf, points_3=fake, id=f"{i}_{classes[int(label_)]}")
        # o3dvis(rawpoints=raw, newpoints=fake, id=f"{i}_{classes[int(label_)]}")

    # print('data_raw.shape:', data_raw.shape)
    # print('data_fake.shape:', data_fake.shape)
    # print('label.shape:', label.shape)








