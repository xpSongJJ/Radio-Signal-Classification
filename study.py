import random
import argparse
import h5py
import torch
from utils import split_data
data_root = "../../data/2018/GOLD_XYZ_OSC.hdf5"

data_test = False
if data_test:
    f = h5py.File(data_root, 'r')
    x, y, z = f['X'], f['Y'], f['Z']
    indexes = []
    '''
    for index, value in enumerate(z):
        if value == 30:
            indexes.append(index)
    print(len(indexes))
    mod_num = 0
    for i in indexes:
        if y[i, 0] == 1:
            mod_num += 1
    print(mod_num)
    '''
    snr_classes = []
    for i in z:
        if i not in snr_classes:
            snr_classes.append(int(i))
    print(snr_classes)

data_type_test = True
if data_type_test:
    f = h5py.File(data_root, 'r')
    x = f['X']
    a = x[0]
    print(a)
    print(a.dtype)

data_y_test = False
if data_y_test:
    with h5py.File(data_root, 'r') as f:
        y = f['Y']
        print(y.shape)
        print(y[0])

list_test = False
if list_test:
    list1 = [1, 2, 3, 4]
    list2 = [5, 6, 7, 8, 9]
    list3 = list1 + list2
    random.seed(42)
    a = random.sample(list3, 5)
    print(a)
    b = random.sample(list3, 5)
    print(b)

check_h5py_data = False
if check_h5py_data:
    data_root = data_root
    train_indexes, train_label, val_indexes, val_label = split_data(data_root, snr=30,
                                                                    ratio=[0.7, 0.1, 0.2], test=False)
    with h5py.File(data_root, 'r') as f:
        x = f['X']  # 信号
        y = f['Y']  # 调制格式
        z = f['Z']  # 信噪比
        num = 0
        for i in range(10):
            print(x[train_indexes[i]])
            print(train_label[i])
            print(y[train_label[i]])


parser_test = False
if parser_test:
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--wd', type=float, default=5e-2)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default="../test_data/")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str,  default="", help='initial weights path')
    # 是否冻结head以外所有权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--snr', type=str, default='30dB')
    opt = parser.parse_args()

    a = opt.num_classes
    b = opt.data_path
    print(a, b)

print(torch.__version__)
