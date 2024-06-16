import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import tensorflow as tf
from scipy.io import loadmat


def _crop():
    num = 4
    num_1 = num + 1
    rate_2 = float('0.' + str(num))
    rate_1 = float('0.' + str(num_1))
    mat = loadmat("SAR_data/California/UiT_HCD_California_2017.mat")
    t1 = tf.convert_to_tensor(mat["t1_L8_clipped"], dtype=tf.float32)
    t2 = tf.convert_to_tensor(mat["logt2_clipped"], dtype=tf.float32)
    change_mask = tf.expand_dims(tf.convert_to_tensor(mat["ROI"], dtype=tf.float32), axis=-1)
    data = tf.concat([t1, t2, change_mask], axis=-1)
    tmp_list = []
    while tmp_list.__len__() < 1:
        tmp = tf.image.random_crop(data, [100, 100, 15])
        rate = tf.reduce_sum(tmp[:, :, 14]) / 10000.0
        if rate >= rate_2:
            if rate <= rate_1:
                tmp_list.append(tmp)
    print("当前的rate是：",rate)
    return tmp_list


def _crop_california():
    num = 0
    num_1 = num+1
    rate_2 = float('0.'+ str(num))
    rate_1 = float('0.' + str(num_1))
    t1 = plt.imread("D:/zhaoqin/BAS_CPU_VER1/Code-Aligned_Autoencoders/data_sample/california_t1.png")
    t2 = plt.imread("D:/zhaoqin/BAS_CPU_VER1/Code-Aligned_Autoencoders/data_sample/california_t2.png")
    change_mask = plt.imread("D:/zhaoqin/BAS_CPU_VER1/Code-Aligned_Autoencoders/data_sample/california_gt.png")
    change_mask = np.expand_dims(change_mask, axis=-1)
    data = np.concatenate((t1, t2, change_mask), axis=-1)
    data = tf.convert_to_tensor(data)
    tmp_list = []
    while tmp_list.__len__() < 1:
        tmp = tf.image.random_crop(data, [25, 25, 7])
        rate = tf.reduce_sum(tmp[:, :, 6]) / 625.0

        if   rate >=rate_2:
            if rate <=rate_1:
                tmp_list.append(tmp)
    print("当前的rate是：",rate)

    x = tmp_list[0][:, :, 0:3]
    y = tmp_list[0][:, :, 3:6]
    gt = tmp_list[0][:, :, 6]

    x_sample = x.numpy()*255
    x_sample = Image.fromarray(x_sample.astype("uint8"))
    x_sample = x_sample.convert("RGB")
    x_sample.save("D:/zhaoqin/BAS_CPU_VER1/Code-Aligned_Autoencoders/data_sample/x_" + str(num) + ".bmp")

    y_sample = y.numpy()*255
    y_sample = Image.fromarray(y_sample.astype("uint8"))
    y_sample = y_sample.convert("RGB")
    y_sample.save("D:/zhaoqin/BAS_CPU_VER1/Code-Aligned_Autoencoders/data_sample/y_" + str(num) + ".bmp")

    gt_sample = gt.numpy()*255
    gt_sample = Image.fromarray(gt_sample.astype("uint8"))
    gt_sample = gt_sample.convert("L")
    gt_sample.save("D:/zhaoqin/BAS_CPU_VER1/Code-Aligned_Autoencoders/data_sample/gt_" + str(num) + ".bmp")
    print('over')
    a = 1



if __name__ == "__main__":

    _crop_california()