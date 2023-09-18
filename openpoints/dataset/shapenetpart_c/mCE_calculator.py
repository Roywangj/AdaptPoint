#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/16 18:56
# @Author  : wangjie
import numpy as np
DGCNN = {
    'clean': 0.8521,
    'scale': [0.8445, 0.8441, 0.8436, 0.8411, 0.8399],
    'jitter': [0.8385, 0.8165, 0.7885, 0.7555, 0.7231],
    'drop_global': [0.8499, 0.8466, 0.8388, 0.8230, 0.7818],
    'drop_local': [0.8456, 0.8403, 0.8316, 0.8160, 0.7991],
    'add_global': [0.6947, 0.6500, 0.6249, 0.6174, 0.6159],
    'add_local': [0.7390, 0.6835, 0.6404, 0.6131, 0.5852],
    'rotate': [0.8453, 0.8246, 0.7934, 0.7577, 0.7154]
}

PointNet = {
    'clean': 0.8331,
    'scale': [0.8310, 0.8314, 0.8287, 0.8296, 0.8280],
    'jitter': [0.8252, 0.8054, 0.7774, 0.7468, 0.7130],
    'drop_global': [0.8330, 0.8320, 0.8310, 0.8300, 0.8290],
    'drop_local': [0.8318, 0.8240, 0.8083, 0.7879, 0.7661],
    'add_global': [0.5910, 0.5263, 0.4908, 0.4621, 0.4393],
    'add_local': [0.6804, 0.6264, 0.5810, 0.5532, 0.5193],
    'rotate': [0.8055, 0.7478, 0.6937, 0.6375, 0.5862]
}

PointNext = {
    'clean': 0.8372,
    'scale': [0.8299, 0.8300, 0.8256, 0.8244, 0.8235],
    'jitter': [0.8230, 0.7917, 0.7620, 0.7205, 0.6729],
    'drop_global': [0.8277, 0.8234, 0.8157, 0.8008, 0.7691],
    'drop_local': [0.8079, 0.7965, 0.7906, 0.7766, 0.7561],
    'add_global': [0.8321, 0.8321, 0.8321, 0.8321, 0.8321],
    'add_local': [0.8315, 0.8301, 0.8323, 0.8320, 0.8326],
    'rotate': [0.8284, 0.8131, 0.7788, 0.6405, 0.6923]
}

PointMAE_adapt = {
    'clean': 0.8523699480893727,
    'scale': [0.8511254345649347, 0.8506446532582723, 0.8491215284908279, 0.8508682747312698,0.8490911869685204],
    'jitter': [0.8399306747299466, 0.8132568366380301, 0.7771399844076747, 0.7366393486411679, 0.7037774805170898],
    'drop_global': [0.8518966814320176, 0.8553470191405397, 0.854583330480289, 0.8554705388558498,0.8561433394509061 ],
    'drop_local': [ 0.852476173494324,  0.8496917243272215, 0.8442001673582271, 0.8338664589412913, 0.8209991996931545],
    'add_global': [0.8523699480893727,0.8523699480893727, 0.8523699480893727, 0.8523699480893727, 0.8523699480893727],
    'add_local': [0.8528035206065798, 0.8515145246147187, 0.8510332402843328,0.8520687032362778,0.85164051735308],
    'rotate': [0.8513532067909144,  0.848954758665358, 0.8399349842172577, 0.8265405907463462, 0.8024973609544334]
}

PointMAE = {
    'clean': 0.8588,
    'scale':       [0.8576, 0.8574, 0.8572, 0.8570, 0.8563],
    'jitter':      [0.8407, 0.8081, 0.7750, 0.7449, 0.7161],
    'drop_global': [0.8578, 0.8584, 0.8562, 0.8517, 0.8432],
    'drop_local':  [0.8584, 0.8564, 0.8525, 0.8408, 0.8272],
    'add_global':  [0.8070, 0.7622, 0.7157, 0.6784, 0.6429],
    'add_local':   [0.7486, 0.6820, 0.6285, 0.5910, 0.5566],
    'rotate':      [0.8498, 0.8277, 0.7951, 0.7537, 0.7067]
}

PointGL = {
    'clean': 0.8509250927721119,
    'scale': [0.847016551140073, 0.8450354203765585, 0.8441813813503132, 0.8451509808932756, 0.8439367256490948],
    'jitter': [0.839627262926785, 0.8085015370267877, 0.7659024855058221, 0.7229404826050276, 0.6787759361417176],
    'drop_global': [0.8473871009373353, 0.8487898428088588, 0.8447805386311636, 0.8373596941153784, 0.8122854047353072],
    'drop_local': [0.8468728441352815, 0.838001190221892, 0.8241841764836112, 0.8008771665573531, 0.7758263446095409],
    'add_global': [0.8509250927721119, 0.8509250927721119, 0.8509250927721119, 0.8509250927721119, 0.8509250927721119],
    'add_local': [0.8512392739421085, 0.8512749677635714, 0.8503636852728513, 0.8502901468636922, 0.8506298123516663],
    'rotate': [0.8484610911855694, 0.8414424393133545, 0.830870613183924, 0.8068368319976402, 0.7760874285055634]
}

PointBert_adapt = {
    'clean': 0.8510185052772827,
    'scale':       [0.847243395346246, 0.8463719069467482, 0.8451571055419832, 0.8469079769971857, 0.8443389642268444],
    'jitter':      [0.839882073003361, 0.8111824635034535, 0.7748224660652685, 0.7356998309422089, 0.6939001513190042],
    'drop_global': [0.8494413537612058, 0.8486040307194619, 0.846425467404258, 0.8401145758511264, 0.8277570796404824],
    'drop_local':  [0.8468851857018455, 0.8397673076575924, 0.831156089589577, 0.8126211314987953, 0.7907803054844494],
    'add_global':  [0.8510185052772827, 0.8510185052772827, 0.8510185052772827, 0.8510185052772827, 0.8510185052772827],
    'add_local':   [0.8505670559227878, 0.8503090571021926, 0.8502045883037597, 0.8512668624662726, 0.8506673127049571],
    'rotate':      [0.8465216956919811, 0.8292677785556472, 0.7978135827533442, 0.7576759359672219, 0.7043582793769024]
}

PCT = {
    'clean': 0.844480464541433,
    'scale':       [0.8400771206035083, 0.8403878129632503, 0.8396205799929087, 0.8371604550262737, 0.83725541175387],
    'jitter':      [0.8319460714048011, 0.7983439985019815, 0.7579214870742838, 0.713872168445319, 0.6684114039768432],
    'drop_global': [0.8409204552222229, 0.8368693081330641, 0.8287310170921133, 0.8098902499825597, 0.7642694479526],
    'drop_local':  [0.8354758420381817, 0.818153721353562, 0.7961251579004667, 0.7682256848511113, 0.7483554417621888,],
    'add_global':  [0.844480464541433, 0.844480464541433, 0.844480464541433, 0.844480464541433, 0.844480464541433,],
    'add_local':   [0.8431456027892602, 0.8450775513177965, 0.8434937426168084, 0.8435822685524483, 0.8432581120609852],
    'rotate':      [0.8415922039231513, 0.8312976377179675, 0.8090960577081027, 0.7807831753250717, 0.7380457647620093]
}

PCT_adaptpoint = {
    'clean': 0.8528053952903701,
    'scale':       [0.8511137926723675, 0.8500318718450327, 0.847619115648758, 0.8474808167553222, 0.8480966440077183],
    'jitter':      [0.843246889511556, 0.8185389360505734, 0.7849184698971914, 0.7523810365181348, 0.7162673237878846],
    'drop_global': [0.8518419595989299, 0.8538821331207517, 0.8524609263493793, 0.8514286786898914, 0.8477782191488492],
    'drop_local':  [0.8483949501287316, 0.8417359947835581, 0.8297962412002134, 0.8128321521683975, 0.7985942245955212],
    'add_global':  [0.8528053952903701, 0.8528053952903701, 0.8528053952903701, 0.8528053952903701, 0.8528053952903701],
    'add_local':   [0.8520475189646421, 0.8530202852987384, 0.8522628019108816, 0.852730459138141, 0.8525871642054914],
    'rotate':      [0.8515667027092876, 0.8487807997454881, 0.8412608899396117, 0.8233664714071799, 0.7980431412413923]
}


def CalculateCE(model):
    scale_DGCNN = np.sum([1 - i for i in DGCNN['scale']])
    jitter_DGCNN = np.sum([1 - i for i in DGCNN['jitter']])
    drop_global_DGCNN = np.sum([1 - i for i in DGCNN['drop_global']])
    drop_local_DGCNN = np.sum([1 - i for i in DGCNN['drop_local']])
    add_global_DGCNN = np.sum([1 - i for i in DGCNN['add_global']])
    add_local_DGCNN = np.sum([1 - i for i in DGCNN['add_local']])
    rotate_DGCNN = np.sum([1 - i for i in DGCNN['rotate']])

    scale = np.sum([1 - i for i in model['scale']])
    jitter = np.sum([1 - i for i in model['jitter']])
    drop_global = np.sum([1 - i for i in model['drop_global']])
    drop_local = np.sum([1 - i for i in model['drop_local']])
    add_global = np.sum([1 - i for i in model['add_global']])
    add_local = np.sum([1 - i for i in model['add_local']])
    rotate = np.sum([1 - i for i in model['rotate']])

    CE_scale = scale / scale_DGCNN
    CE_jitter = jitter / jitter_DGCNN
    CE_drop_global = drop_global / drop_global_DGCNN
    CE_drop_local = drop_local / drop_local_DGCNN
    CE_add_global = add_global / add_global_DGCNN
    CE_add_local = add_local / add_local_DGCNN
    CE_rotate = rotate / rotate_DGCNN

    mCE = (CE_scale + CE_jitter + CE_drop_global + CE_drop_local + CE_add_global + CE_add_local + CE_rotate) / 7

    print("mCE: {:.3f}".format(mCE))
    print("CE (scale): {:.3f}".format(CE_scale))
    print("CE (jitter): {:.3f}".format(CE_jitter))
    print("CE (drop_global): {:.3f}".format(CE_drop_global))
    print("CE (drop_local): {:.3f}".format(CE_drop_local))
    print("CE (add_global): {:.3f}".format(CE_add_global))
    print("CE (add_local): {:.3f}".format(CE_add_local))
    print("CE (rotate): {:.3f}".format(CE_rotate))


def CalculateRCE(model):
    scale_DGCNN = np.sum([DGCNN['clean'] - i for i in DGCNN['scale']])
    jitter_DGCNN = np.sum([DGCNN['clean'] - i for i in DGCNN['jitter']])
    drop_global_DGCNN = np.sum([DGCNN['clean'] - i for i in DGCNN['drop_global']])
    drop_local_DGCNN = np.sum([DGCNN['clean'] - i for i in DGCNN['drop_local']])
    add_global_DGCNN = np.sum([DGCNN['clean'] - i for i in DGCNN['add_global']])
    add_local_DGCNN = np.sum([DGCNN['clean'] - i for i in DGCNN['add_local']])
    rotate_DGCNN = np.sum([DGCNN['clean'] - i for i in DGCNN['rotate']])

    scale = np.sum([model['clean'] - i for i in model['scale']])
    jitter = np.sum([model['clean'] - i for i in model['jitter']])
    drop_global = np.sum([model['clean'] - i for i in model['drop_global']])
    drop_local = np.sum([model['clean'] - i for i in model['drop_local']])
    add_global = np.sum([model['clean'] - i for i in model['add_global']])
    add_local = np.sum([model['clean'] - i for i in model['add_local']])
    rotate = np.sum([model['clean'] - i for i in model['rotate']])

    RCE_scale = scale / scale_DGCNN
    RCE_jitter = jitter / jitter_DGCNN
    RCE_drop_global = drop_global / drop_global_DGCNN
    RCE_drop_local = drop_local / drop_local_DGCNN
    RCE_add_global = add_global / add_global_DGCNN
    RCE_add_local = add_local / add_local_DGCNN
    RCE_rotate = rotate / rotate_DGCNN

    mRCE = (RCE_scale + RCE_jitter + RCE_drop_global + RCE_drop_local + RCE_add_global + RCE_add_local + RCE_rotate) / 7

    print("mRCE: {:.3f}".format(mRCE))
    print("RCE (scale): {:.3f}".format(RCE_scale))
    print("RCE (jitter): {:.3f}".format(RCE_jitter))
    print("RCE (drop_global): {:.3f}".format(RCE_drop_global))
    print("RCE (drop_local): {:.3f}".format(RCE_drop_local))
    print("RCE (add_global): {:.3f}".format(RCE_add_global))
    print("RCE (add_local): {:.3f}".format(RCE_add_local))
    print("RCE (rotate): {:.3f}".format(RCE_rotate))
def main():
    CalculateCE(PointGL)
    CalculateRCE(PointGL)

    # data_dict = transdata2dict('79.67	55.645	54.143	36.086	63.137	53.72	67.495	43.227	71.707')


if __name__ == '__main__':
    print('==>begining...')
    main()
    print('==>ending...')
