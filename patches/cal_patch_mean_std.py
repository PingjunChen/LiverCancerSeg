# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io
import pydaily

data_root = "../data/Patches"
tumor_type = "whole"
train_dir = os.path.join(data_root, tumor_type, 'train')
val_dir = os.path.join(data_root, tumor_type, 'val')


def get_mean_and_std(img_dir, suffix):
    mean, std = np.zeros(3), np.zeros(3)
    filelist = pydaily.filesystem.find_ext_files(img_dir, suffix)

    for idx, filepath in enumerate(filelist):
        cur_img = io.imread(filepath) / 255.0
        for i in range(3):
            mean[i] += np.mean(cur_img[:,:,i])
            std[i] += cur_img[:,:,i].std()
    mean = [ele * 1.0 / len(filelist) for ele in mean]
    std = [ele * 1.0 / len(filelist) for ele in std]
    return mean, std

rgb_mean, rgb_std = get_mean_and_std(os.path.join(train_dir, "imgs"), suffix=".jpg")
print("mean rgb: {}".format(rgb_mean))
print("std rgb: {}".format(rgb_std))
