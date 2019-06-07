# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt

from pydaily import filesystem
import tissueloc as tl
from pyslide import pyramid
from pycontour.cv2_transform import cv_cnt_to_np_arr, np_arr_to_cv_cnt
from pycontour.poly_transform import np_arr_to_poly, poly_to_np_arr


def locate_tissue(slides_dir):
    slide_list = []
    svs_file_list = filesystem.find_ext_files(slides_dir, "svs")
    slide_list.extend(svs_file_list)
    SVS_file_list = filesystem.find_ext_files(slides_dir, "SVS")
    slide_list.extend(SVS_file_list)

    tissue_dir = os.path.join(os.path.dirname(slides_dir), "Visualization/TissueLoc")
    filesystem.overwrite_dir(tissue_dir)
    for ind, slide_path in enumerate(slide_list):
        print("processing {}/{}".format(ind+1, len(slide_list)))
        # locate tissue contours with default parameters
        cnts, d_factor = tl.locate_tissue_cnts(slide_path, max_img_size=2048, smooth_sigma=13,
                                               thresh_val=0.88, min_tissue_size=120000)
        cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)

        # if len(cnts) != 1:
        #     print("There are {} contours in {}".format(len(cnts), os.path.basename(slide_path)))

        # load slide
        select_level, select_factor = tl.select_slide_level(slide_path, max_size=2048)
        wsi_head = pyramid.load_wsi_head(slide_path)
        slide_img = wsi_head.read_region((0, 0), select_level, wsi_head.level_dimensions[select_level])
        slide_img = np.asarray(slide_img)[:,:,:3]
        slide_img = np.ascontiguousarray(slide_img, dtype=np.uint8)

        # change not valid poly to convex_hull
        cnt_arr = cv_cnt_to_np_arr(cnts[0])
        cnt_poly = np_arr_to_poly(cnt_arr)
        if cnt_poly.is_valid == True:
            valid_cnt = cnts[0].astype(int)
        else:
            valid_arr = poly_to_np_arr(cnt_poly.convex_hull)
            valid_cnt = np_arr_to_cv_cnt(valid_arr).astype(int)
        cv2.drawContours(slide_img, [valid_cnt], 0, (0, 255, 0), 8)

        # overlay and save
        # cv2.drawContours(slide_img, cnts, 0, (0, 255, 0), 8)
        tissue_save_name = os.path.splitext(os.path.basename(slide_path))[0] + ".png"
        tissue_save_path = os.path.join(tissue_dir, tissue_save_name)
        io.imsave(tissue_save_path, slide_img)


if __name__ == "__main__":
    slides_dir = os.path.join("../data", "LiverImages")
    locate_tissue(slides_dir)
