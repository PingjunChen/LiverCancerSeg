# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pydaily import filesystem


def get_slide_filenames(slides_dir):
    slide_list = []
    svs_file_list = filesystem.find_ext_files(slides_dir, "svs")
    slide_list.extend([os.path.basename(ele) for ele in svs_file_list])
    SVS_file_list = filesystem.find_ext_files(slides_dir, "SVS")
    slide_list.extend([os.path.basename(ele) for ele in SVS_file_list])
    slide_filenames = [os.path.splitext(ele)[0] for ele in slide_list]

    slide_filenames.sort()

    return slide_filenames


def mask2color(mask):
    # colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56),
    #                      (0, 0, 117), (128, 128, 0), (191, 239, 69), (145, 30, 180)])
    color_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color_img[mask==255] = (191, 239, 69)

    return color_img


def gen_patch_pred(inputs, masks, preds):
    imgs = inputs.permute(0, 2, 3, 1).data.cpu().numpy()
    masks = torch.squeeze(masks, dim=1).data.cpu().numpy()
    preds = torch.squeeze(F.sigmoid(preds), dim=1).data.cpu().numpy()
    imgs = (imgs * 255).astype(np.uint8)
    masks = ((masks > 0.5) * 255).astype(np.uint8)
    preds = ((preds > 0.5) * 255).astype(np.uint8)

    img_num, img_size = imgs.shape[0], imgs.shape[1]
    result_img = np.zeros((img_num*img_size, img_size*3, imgs.shape[3]), dtype=np.uint8)
    for ind in np.arange(img_num):
        result_img[ind*img_size:(ind+1)*img_size, :img_size] = imgs[ind]
        result_img[ind*img_size:(ind+1)*img_size, img_size:img_size*2] = mask2color(masks[ind])
        result_img[ind*img_size:(ind+1)*img_size, img_size*2:img_size*3] = mask2color(preds[ind])

    return result_img


def gen_patch_mask_wmap(slide_img, mask_img, coors_arr, plen):
    patch_list, mask_list = [], []
    wmap = np.zeros((slide_img.shape[0], slide_img.shape[1]), dtype=np.int32)
    for coor in coors_arr:
        ph, pw = coor[0], coor[1]
        patch_list.append(slide_img[ph:ph+plen, pw:pw+plen] / 255.0)
        mask_list.append(mask_img[ph:ph+plen, pw:pw+plen])
        wmap[ph:ph+plen, pw:pw+plen] += 1
    patch_arr = np.asarray(patch_list).astype(np.float32)
    mask_arr = np.asarray(mask_list).astype(np.float32)

    return patch_arr, mask_arr, wmap


def gen_patch_wmap(slide_img, coors_arr, plen):
    patch_list = []
    wmap = np.zeros((slide_img.shape[0], slide_img.shape[1]), dtype=np.int32)
    for coor in coors_arr:
        ph, pw = coor[0], coor[1]
        patch_list.append(slide_img[ph:ph+plen, pw:pw+plen] / 255.0)
        wmap[ph:ph+plen, pw:pw+plen] += 1
    patch_arr = np.asarray(patch_list).astype(np.float32)

    return patch_arr, wmap


def wsi_stride_splitting(wsi_h, wsi_w, patch_len, stride_len):
    """ Spltting whole slide image to patches by stride.

    Parameters
    -------
    wsi_h: int
        height of whole slide image
    wsi_w: int
        width of whole slide image
    patch_len: int
        length of the patch image
    stride_len: int
        length of the stride

    Returns
    -------
    coors_arr: list
        list of starting coordinates of patches ([0]-h, [1]-w)

    """

    coors_arr = []
    def stride_split(ttl_len, patch_len, stride_len):
        p_sets = []
        if patch_len > ttl_len:
            raise AssertionError("patch length larger than total length")
        elif patch_len == ttl_len:
            p_sets.append(0)
        else:
            stride_num = int(np.ceil((ttl_len - patch_len) * 1.0 / stride_len))
            for ind in range(stride_num+1):
                cur_pos = int(((ttl_len - patch_len) * 1.0 / stride_num) * ind)
                p_sets.append(cur_pos)

        return p_sets

    h_sets = stride_split(wsi_h, patch_len, stride_len)
    w_sets = stride_split(wsi_w, patch_len, stride_len)

    # combine points in both w and h direction
    if len(w_sets) > 0 and len(h_sets) > 0:
        coors_arr = list(itertools.product(h_sets, w_sets))

    return coors_arr


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
