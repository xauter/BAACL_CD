import os

# Set loglevel to suppress tensorflow GPU messages
# import dtcwt

# from upsample import double_linear

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
import re
from itertools import count

import numpy as np
import tensorflow as tf
from scipy.io import loadmat, savemat
from change_priors import eval_prior, remove_borders, image_in_patches

def _california(reduce=False):
    """ Load California dataset from .mat """
    mat = loadmat("data/California/UiT_HCD_California_2017.mat")

    t1 = tf.convert_to_tensor(mat["t1_L8_clipped"], dtype=tf.float32)
    t2 = tf.convert_to_tensor(mat["logt2_clipped"], dtype=tf.float32)
    change_mask = tf.convert_to_tensor(mat["ROI"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    if reduce:
        print("Reducing")
        reduction_ratios = (4, 4)
        new_dims = list(map(lambda a, b: a // b, change_mask.shape, reduction_ratios))
        t1 = tf.cast(tf.image.resize(t1, new_dims, antialias=True), dtype=tf.float32)
        t2 = tf.cast(tf.image.resize(t2, new_dims, antialias=True), dtype=tf.float32)
        change_mask = tf.cast(
            tf.image.resize(tf.cast(change_mask, tf.uint8), new_dims, antialias=True),
            tf.bool,
        )

    return t1, t2, change_mask


def _texas(clip=True):
    """ Load Texas dataset from .mat """
    mat = loadmat("data/Texas/Cross-sensor-Bastrop-data.mat")

    t1 = np.array(mat["t1_L5"], dtype=np.single)
    t2 = np.array(mat["t2_ALI"], dtype=np.single)
    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(mat["ROI_1"], dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]

    return t1, t2, change_mask

def _xidian(clip=True):
    t1 = plt.imread("./data/xidian/1.bmp")
    t2 = plt.imread("./data/xidian/2.bmp")
    change_mask = plt.imread("./data/xidian/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _xidian2(clip=True):
    t1 = plt.imread("./data/xidian2/1.bmp")
    t2 = plt.imread("./data/xidian2/2.bmp")
    change_mask = plt.imread("./data/xidian2/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _ychang(clip=True):
    t1 = plt.imread("./data/ychang/1.bmp")
    t2 = plt.imread("./data/ychang/2.bmp")
    change_mask = plt.imread("./data/ychang/3.bmp")
    # t1 = t1[:, :, 0]
    change_mask = change_mask[:, :, 0]
    # t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _bern(clip=True):
    t1 = plt.imread("./data/Bern/1.bmp")
    t2 = plt.imread("./data/Bern/2.bmp")
    change_mask = plt.imread("./data/Bern/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :, 0]
    t2 = t2[:, :, 0]
    t2 = t2[..., np.newaxis]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _air(clip=True):
    t1 = plt.imread("./data/air/1.bmp")
    t2 = plt.imread("./data/air/2.bmp")
    change_mask = plt.imread("./data/air/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :, 0]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _square(clip=True):
    t1 = plt.imread("./data/square/1.bmp")
    t2 = plt.imread("./data/square/2.bmp")
    change_mask = plt.imread("./data/square/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :, 0]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask

def _italy(clip=True):
    t1 = plt.imread("./data/Italy/1.bmp")
    t2 = plt.imread("./data/Italy/2.bmp")
    change_mask = plt.imread("./data/Italy/3.bmp")
    t1 = t1[:, :, 0]
    change_mask = change_mask[:, :, 0]
    t1 = t1[..., np.newaxis]

    t1 = np.array(t1, dtype=np.single)
    t2 = np.array(t2, dtype=np.single)


    if clip:
        print("clipping")
        t1, t2 = _clip(t1), _clip(t2)
    change_mask = tf.convert_to_tensor(change_mask,dtype=tf.bool)
    assert t1.shape[:2] == t2.shape[:2] == change_mask.shape[:2]
    if change_mask.ndim == 2:
        change_mask = change_mask[..., np.newaxis]
    return t1, t2, change_mask


def _clip(image):
    """
        Normalize image from R_+ to [-1, 1].

        For each channel, clip any value larger than mu + 3sigma,
        where mu and sigma are the channel mean and standard deviation.
        Scale to [-1, 1] by (2*pixel value)/(max(channel)) - 1

        Input:
            image - (h, w, c) image array in R_+
        Output:
            image - (h, w, c) image array normalized within [-1, 1]
    """

    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))

    return image


def _training_data_generator(x, y, gt, p, patch_size):
    """
        Factory for generator used to produce training dataset.
        The generator will choose a random patch and flip/rotate the images

        Input:
            x - tensor (h, w, c_x)
            y - tensor (h, w, c_y)
            p - tensor (h, w, 1)
            patch_size - int in [1, min(h,w)], the size of the square patches
                         that are extracted for each training sample.
        Output:
            to be used with tf.data.Dataset.from_generator():
                gen - generator callable yielding
                    x - tensor (ps, ps, c_x)
                    y - tensor (ps, ps, c_y)
                    p - tensor (ps, ps, 1)
                dtypes - tuple of tf.dtypes
                shapes - tuple of tf.TensorShape
    """
    gt = np.array(gt, dtype=np.float)
    gt = tf.convert_to_tensor(gt, dtype=tf.float32)
    c_x, c_y, c_gt= x.shape[2], y.shape[2], gt.shape[2]
    chs = c_x + c_y + c_gt + 1
    x_chs = slice(0, c_x, 1)
    y_chs = slice(c_x, c_x + c_y, 1)
    p_chs = slice(c_x + c_y, c_x + c_y + 1, 1)
    gt_chs = slice(c_x + c_y + 1, chs, 1)
    data = tf.concat([x, y, p, gt], axis=-1)
    #将gt放在最后一维上
    #此处的data是一个300*412*5的张量，在最后一维上将x,y,p(此处p全0.指的是交换损失权重cross_loss_weight)
    def gen():
        for _ in count():
            tmp = tf.image.random_crop(data, [patch_size, patch_size, chs])
            # tmp = tf.image.rot90(tmp, np.random.randint(4))
            # tmp = tf.image.random_flip_up_down(tmp)

            yield tmp[:, :, x_chs], tmp[:, :, y_chs], tmp[:, :, p_chs], tmp[:, :, gt_chs]

    dtypes = (tf.float32, tf.float32, tf.float32, tf.float32)
    shapes = (
        tf.TensorShape([patch_size, patch_size, c_x]),
        tf.TensorShape([patch_size, patch_size, c_y]),
        tf.TensorShape([patch_size, patch_size, 1]),
        tf.TensorShape([patch_size, patch_size, c_gt])
    )

    return gen, dtypes, shapes


DATASETS = {
    "xidian2": _xidian2,
    "California": _california,
    "Italy": _italy,
    "Air": _air,
}
prepare_data = {
    "xidian2": True,
    "California": True,
    "Italy": True,
    "Air": True,
}


def fetch(name, patch_size=100, **kwargs):
    """
        Input:
            name - dataset name, should be in DATASETS
            kwargs - config {key: value} pairs.
                     Key should be in DATASET_DEFAULT_CONFIG
        Output:
            training_data - tf.data.Dataset with (x, y, prior)
                            shapes like (inf, patch_size, patch_size, ?)
            evaluation_data - tf.data.Dataset with (x, y, change_map)
                              shapes (1, h, w, ?)
            channels - tuple (c_x, c_y), number of channels for domains x and y
    """
    x_im, y_im, target_cm = DATASETS[name](prepare_data[name])

    if not tf.config.list_physical_devices("GPU"):    # default: if not tf.config.list_physical_devices("GPU"):
        dataset = [
            tf.image.central_crop(tensor, 0.1) for tensor in [x_im, y_im, target_cm]
        ]
    else:
        dataset = [x_im, y_im, target_cm]

    dataset = [tf.expand_dims(tensor, 0) for tensor in dataset]
    x, y ,target_cm= dataset[0], dataset[1], dataset[2]
    evaluation_data = tf.data.Dataset.from_tensor_slices(tuple(dataset))

    c_x, c_y = x_im.shape[-1], y_im.shape[-1]

    return x, y, target_cm, evaluation_data, (c_x, c_y)


if __name__ == "__main__":
    for DATASET in DATASETS:
        print(f"Loading {DATASET}")
        fetch(DATASET)
