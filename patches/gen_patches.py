# -*- coding: utf-8 -*-

import os, sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import uuid, shutil
from skimage import io, transform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from pydaily import filesystem
import tissueloc as tl
from pyslide import pyramid, contour, patch
from pycontour.cv2_transform import cv_cnt_to_np_arr
from pycontour.poly_transform import np_arr_to_poly, poly_to_np_arr


def gen_samples(slides_dir, patch_level, patch_size, tumor_type, slide_list, dset, overlap_mode):
    # prepare saving directory
    patch_path = os.path.join(os.path.dirname(slides_dir), "Patches", tumor_type)
    patch_img_dir = os.path.join(patch_path, dset, "imgs")
    if not os.path.exists(patch_img_dir):
        os.makedirs(patch_img_dir)
    patch_mask_dir = os.path.join(patch_path, dset, "masks")
    if not os.path.exists(patch_mask_dir):
        os.makedirs(patch_mask_dir)

    # processing slide one-by-one
    ttl_patch = 0
    slide_list.sort()
    for ind, ele in enumerate(slide_list):
        print("Processing {} {}/{}".format(ele, ind+1, len(slide_list)))
        cur_slide_path = os.path.join(slides_dir, ele+".svs")
        if os.path.exists(cur_slide_path):
            cur_slide_path = os.path.join(slides_dir, ele+".svs")

        # locate contours and generate batches based on tissue contours
        cnts, d_factor = tl.locate_tissue_cnts(cur_slide_path, max_img_size=2048, smooth_sigma=13,
                                               thresh_val=0.88, min_tissue_size=120000)
        select_level, select_factor = tl.select_slide_level(cur_slide_path, max_size=2048)
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)

        # scale contour to slide level 2
        wsi_head = pyramid.load_wsi_head(cur_slide_path)
        cnt_scale = select_factor / int(wsi_head.level_downsamples[patch_level])
        tissue_arr = cv_cnt_to_np_arr(cnts[0] * cnt_scale).astype(np.int32)
        # convert tissue_arr to convex if poly is not valid
        tissue_poly = np_arr_to_poly(tissue_arr)
        if tissue_poly.is_valid == False:
            tissue_arr = poly_to_np_arr(tissue_poly.convex_hull).astype(int)

        coors_arr = None
        if overlap_mode == "half_overlap":
            level_w, level_h = wsi_head.level_dimensions[patch_level]
            coors_arr = contour.contour_patch_splitting_half_overlap(tissue_arr, level_h, level_w, patch_size, inside_ratio=0.80)
        elif overlap_mode == "self_overlap":
            coors_arr = contour.contour_patch_splitting_self_overlap(tissue_arr, patch_size, inside_ratio=0.80)
        else:
            raise NotImplementedError("unknown overlapping mode")

        wsi_img = wsi_head.read_region((0, 0), patch_level, wsi_head.level_dimensions[patch_level])
        wsi_img = np.asarray(wsi_img)[:,:,:3]
        mask_path = os.path.join(slides_dir, "_".join([ele, tumor_type+".tif"]))
        mask_img = io.imread(mask_path)
        wsi_mask = (transform.resize(mask_img, wsi_img.shape[:2], order=0) * 255).astype(np.uint8) * 255

        if dset == "val":
            test_slides_dir = os.path.join(os.path.dirname(slides_dir), "TestSlides")
            if not os.path.exists(os.path.join(test_slides_dir, cur_slide_path)):
                shutil.copy(cur_slide_path, test_slides_dir)
            if not os.path.exists(os.path.join(test_slides_dir, mask_path)):
                shutil.copy(mask_path, test_slides_dir)

        for cur_arr in coors_arr:
            cur_h, cur_w = cur_arr[0], cur_arr[1]
            cur_patch = wsi_img[cur_h:cur_h+patch_size, cur_w:cur_w+patch_size]
            if cur_patch.shape[0] != patch_size or cur_patch.shape[1] != patch_size:
                continue
            cur_mask = wsi_mask[cur_h:cur_h+patch_size, cur_w:cur_w+patch_size]
            # background RGB (235, 210, 235) * [0.299, 0.587, 0.114]
            if patch.patch_bk_ratio(cur_patch, bk_thresh=0.864) > 0.88:
                continue

            if overlap_mode == "half_overlap" and tumor_type == "viable":
                pixel_ratio = np.sum(cur_mask > 0) * 1.0 / cur_mask.size
                if pixel_ratio < 0.05:
                    continue

            patch_name = ele + "_" + str(uuid.uuid1())[:8]
            io.imsave(os.path.join(patch_img_dir, patch_name+".jpg"), cur_patch)
            io.imsave(os.path.join(patch_mask_dir, patch_name+".png"), cur_mask)
            ttl_patch += 1

    print("There are {} patches in total.".format(ttl_patch))



if __name__ == "__main__":
    # prepare train and validation slide list
    mask_dir = os.path.join("../data", "Visualization", "TissueLoc")
    slide_list = [os.path.splitext(ele)[0] for ele in os.listdir(mask_dir) if "png" in ele]
    train_slide_list, val_slide_list = train_test_split(slide_list, test_size=0.20, random_state=1234)

    # generate patches for segmentation model training
    slides_dir = os.path.join("../data", "LiverImages")
    patch_level, patch_size = 2, 512
    # tumor_type = "viable"
    tumor_types = ["viable", "whole"]
    for cur_type in tumor_types:
        print("Generating {} tumor patches.".format(cur_type))
        patch_modes = [(val_slide_list, "val", "half_overlap"), (val_slide_list, "val", "self_overlap"),
                       (train_slide_list, "train", "half_overlap"), (train_slide_list, "train", "self_overlap")]
        for mode in patch_modes:
            gen_samples(slides_dir, patch_level, patch_size, cur_type, mode[0], mode[1], overlap_mode=mode[2])
