"""
    dataset of shapenetpart-c
"""

import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import argparse
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import sys
# sys.path.append("../data_utils")

# print(f'sys.path:{sys.path}')
# print(os.path.abspath('.') )
# print(os.path.abspath('..') )
# sys.path.append('..')
# print(f'sys.path:{sys.path}')

def load_data_partseg(partition, sub=None):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../data')
    SHAPENET_DIR = os.path.join(DATA_DIR, 'hdf5_data')
    SHAPENET_C_DIR = os.path.join(DATA_DIR, 'shapenet_c')

    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(SHAPENET_DIR, '*train*.h5')) \
               + glob.glob(os.path.join(SHAPENET_DIR,  '*val*.h5'))
        # file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*train*.h5')) \
        #        + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*val*.h5'))
    elif partition == 'shapenet-c':
        file = os.path.join(SHAPENET_C_DIR, '%s.h5'%sub)
    else:
        file = glob.glob(os.path.join(SHAPENET_DIR, '*%s*.h5'%partition))
        # file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*%s*.h5'%partition))

    if partition == 'shapenet-c':
    # for h5_name in file:
        # f = h5py.File(h5_name, 'r+')
        f = h5py.File(file, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')  # part seg label
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    else:
        for h5_name in file:
            f = h5py.File(h5_name, 'r')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            seg = f['pid'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            all_seg.append(seg)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg




def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ShapeNetPart(Dataset):
    def __init__(self, num_points=2048, partition='train', class_choice=None, sub=None):
        self.data, self.label, self.seg = load_data_partseg(partition, sub)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition
        self.class_choice = class_choice

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

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]  # part seg label
        if self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]



class ShapeNetC(Dataset):
    def __init__(self, partition='train', class_choice=None, sub=None):
        self.data, self.label, self.seg = load_data_partseg(partition, sub)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4,
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9,
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]  # number of parts for each category
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.partition = partition
        self.class_choice = class_choice
        # self.partseg_colors = load_color_partseg()

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

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        seg = self.seg[item]  # part seg label
        if self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]



import pprint
def eval_corrupt_wrapper_epoch(model, fn_test_corrupt, args_test_corrupt, path, epoch):
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


    # for k in perf_all:
    #     perf_all[k] = sum(perf_all[k]) / len(perf_all[k])
    #     perf_all[k] = round(perf_all[k], 3)

    #     if corruption_type != 'clean':
    #         perf_corrupt['CE'] = (1 - perf_corrupt['OA']) / (1 - DGCNN_OA[corruption_type])
    #         perf_corrupt['RCE'] = (OA_clean - perf_corrupt['OA']) / (DGCNN_OA['clean'] - DGCNN_OA[corruption_type])
    #         for k in perf_all:
    #             perf_corrupt[k] = round(perf_corrupt[k], 3)
    #             perf_all[k].append(perf_corrupt[k])
    #     perf_corrupt['corruption'] = corruption_type
    #     perf_corrupt['level'] = 'Overall'
    #     pprint.pprint(perf_corrupt, width=200)
    #     file.write(f"{perf_corrupt} \n")
    # for k in perf_all:
    #     perf_all[k] = sum(perf_all[k]) / len(perf_all[k])
    #     perf_all[k] = round(perf_all[k], 3)
    # perf_all['mCE'] = perf_all.pop('CE')
    # perf_all['RmCE'] = perf_all.pop('RCE')
    # perf_all['mOA'] = perf_all.pop('OA')
    # pprint.pprint(perf_all, width=200)
    # file.write(f"{perf_all} \n")
    # file.close()



if __name__ == '__main__':

    # shapenet_train = ShapeNetPart()
    split='clean'
    # shapenet_train = ShapeNetC(partition='trainval', sub=split, class_choice=None)
    shapenet_train = ShapeNetC(partition='shapenet-c', sub=split, class_choice=None)
    for id in range(1):
        pointcloud, label, seg = shapenet_train.__getitem__(id)
        print(pointcloud.shape)
        print(label)
    print("==> ending...")

