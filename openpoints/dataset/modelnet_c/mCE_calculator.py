#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/5 15:59
# @Author  : wangjie
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 16:45
# @Author  : wangjie

import numpy as np
import re

corruptions = [
    'scale',
    'jitter',
    'rotate',
    'dropout_global',
    'dropout_local',
    'add_global',
    'add_local',
]

DGCNN = {
    'clean': 0.926,
    'scale': 0.906,
    'jitter': 0.684,
    'rotate': 0.785,
    'dropout_global': 0.752,
    'dropout_local': 0.793,
    'add_global': 0.705,
    'add_local': 0.725
}

PointNet2_wwolfmix = {'clean': 0.931, 'scale': 0.911, 'jitter': 0.567, 'rotate': 0.891, 'dropout_global': 0.886, 'dropout_local': 0.873, 'add_global': 0.912, 'add_local': 0.919}

def CalculateCE(model):
    perf_all = {'CE': [], 'RCE': []}
    for corruption_type in corruptions:
        perf_corruption = {'CE': [], 'RCE': []}
        CE = (1 - model[corruption_type]) / (1 - DGCNN[corruption_type])
        RCE = (model['clean'] - model[corruption_type]) / (DGCNN['clean'] - DGCNN[corruption_type])
        perf_corruption['CE'].append(round(CE, 3))
        perf_corruption['RCE'].append(round(RCE, 3))
        perf_corruption['corruption'] = corruption_type
        perf_corruption['level'] = 'Overall'
        print(f'perf_corruption:{perf_corruption}')
        perf_all['CE'].append(CE)
        perf_all['RCE'].append(RCE)
    print(f'perf_all before average:{perf_all}')
    for k in perf_all:
        perf_all[k] = sum(perf_all[k]) / len(perf_all[k])
        perf_all[k] = round(perf_all[k], 3)
    perf_all['mCE'] = perf_all.pop('CE')
    perf_all['RmCE'] = perf_all.pop('RCE')
    # print(f'perf_corruption:{perf_corruption}')
    print(f'perf_all:{perf_all}')
    pass

def transdata2dict(string):
    data_list = re.findall(r"\d+\.?\d*", string)
    for i in range(len(data_list)):
        data_list[i] = round((float(data_list[i]) / 100), 5)
    data_dict={
    'clean': data_list[0],
    'scale': data_list[2],
    'jitter': data_list[3],
    'rotate': data_list[4],
    'dropout_global': data_list[5],
    'dropout_local': data_list[6],
    'add_global': data_list[7],
    'add_local': data_list[8]
    }
    print(data_dict)
    return data_dict



def main():
    CalculateCE(PointNet2_wwolfmix)
    # data_dict = transdata2dict('87.47	67.191	61.291	45.517	71.541	80.049	80.298	52.033	79.611')


if __name__ == '__main__':
    print('==>begining...')
    main()
    print('==>ending...')
