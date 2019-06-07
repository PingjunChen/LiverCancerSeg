# -*- coding: utf-8 -*-

import os, sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import uuid, shutil
from skimage import io, transform
from sklearn.model_selection import train_test_split

import tissueloc as tl
from pydaily import filesystem
from pyslide import pyramid, contour, patch
from pycontour.cv2_transform import cv_cnt_to_np_arr
from pycontour.poly_transform import np_arr_to_poly, poly_to_np_arr


def gen_seg_samples(slides_dir, slide_list, dset, patch_level, patch_size):
    # prepare saving directory
    patch_path = os.path.join(os.path.dirname(slides_dir), "Patches")
    patch_img_dir = os.path.join(patch_path, dset, "imgs")
    filesystem.overwrite_dir(patch_img_dir)
    patch_mask_dir = os.path.join(patch_path, dset, "masks")
    filesystem.overwrite_dir(patch_mask_dir)

    if dset == "val":
        test_slides_dir = os.path.join(os.path.dirname(slides_dir), "TestSlides")
        filesystem.overwrite_dir(test_slides_dir)

    # processing slide one-by-one
    ttl_patch = 0
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

        coors_arr = contour.contour_patch_splitting_self_overlap(tissue_arr, patch_size, inside_ratio=0.80)
        # level_w, level_h = wsi_head.level_dimensions[patch_level]
        # coors_arr = contour.contour_patch_splitting_half_overlap(tissue_arr, level_h, level_w, patch_size, inside_ratio=0.80)

        wsi_img = wsi_head.read_region((0, 0), patch_level, wsi_head.level_dimensions[patch_level])
        wsi_img = np.asarray(wsi_img)[:,:,:3]
        viable_mask_path = os.path.join(slides_dir, ele+"_viable.tif")
        viable_mask_img = io.imread(viable_mask_path)
        wsi_mask = (transform.resize(viable_mask_img, wsi_img.shape[:2], order=0) * 255).astype(np.uint8) * 255
        # if dset == "val":
        #     shutil.copy(cur_slide_path, test_slides_dir)
        #     shutil.copy(viable_mask_path, test_slides_dir)

        for cur_arr in coors_arr:
            cur_h, cur_w = cur_arr[0], cur_arr[1]
            cur_patch = wsi_img[cur_h:cur_h+patch_size, cur_w:cur_w+patch_size]
            if cur_patch.shape[0] != patch_size or cur_patch.shape[1] != patch_size:
                continue
            # background RGB (235, 210, 235) * [0.299, 0.587, 0.114]
            if patch.patch_bk_ratio(cur_patch, bk_thresh=0.864) > 0.8:
                continue
            cur_mask = wsi_mask[cur_h:cur_h+patch_size, cur_w:cur_w+patch_size]
            pixel_ratio = np.sum(cur_mask > 0) * 1.0 / cur_mask.size
            if pixel_ratio < 0.05:
                continue
            patch_name = ele + "_" + str(uuid.uuid1())[:8]
            io.imsave(os.path.join(patch_img_dir, patch_name+".jpg"), cur_patch)
            io.imsave(os.path.join(patch_mask_dir, patch_name+".png"), cur_mask)
            ttl_patch += 1

    print("There are {} patches in total.".format(ttl_patch))



if __name__ == "__main__":
    np.random.seed(1234)
    # prepare train and validation slide list
    mask_dir = os.path.join("../data", "Visualization", "TissueLoc")
    slide_list = [os.path.splitext(ele)[0] for ele in os.listdir(mask_dir) if "png" in ele]
    train_slide_list, val_slide_list = train_test_split(slide_list, test_size=0.20, random_state=1234)

    # generate patches for segmentation model training
    slides_dir = os.path.join("../data", "LiverImages")
    patch_level, patch_size = 2, 512
    # generate validation patch samples
    gen_seg_samples(slides_dir, val_slide_list, "val", patch_level, patch_size)
    # generate training patch samples
    gen_seg_samples(slides_dir, train_slide_list, "train", patch_level, patch_size)
