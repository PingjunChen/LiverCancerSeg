# -*- coding: utf-8 -*-

import os, sys
import numpy as np

from pydaily import filesystem
from pyslide import pyramid


def get_slide_list(slides_dir):
    slide_list = []
    svs_file_list = filesystem.find_ext_files(slides_dir, "svs")
    slide_list.extend(svs_file_list)
    SVS_file_list = filesystem.find_ext_files(slides_dir, "SVS")
    slide_list.extend(SVS_file_list)

    return slide_list


def check_slide_properties(slide_path):
    wsi_head = pyramid.load_wsi_head(slide_path)
    flag = True
    # if wsi_head.level_count <= 2:
    #     print("{} has {} levels".format(wsi_head._filename, wsi_head.level_count))
    #     flag = False
    # print(wsi_head.level_downsamples)
    if np.absolute(wsi_head.level_downsamples[2] - 16) > 0.01:
        print("{} scale is not {}".format(wsi_head._filename, 4))
        flag = False

    return flag


def check_slide_level(slide_list):
    fail_num = 0
    for index, slide_path in enumerate(slide_list):
        check_flag = check_slide_properties(slide_path)
        if check_flag == False:
            fail_num += 1
    print("There are {} slides not satisfying properties.".format(fail_num))


if __name__ == "__main__":
    slides_dir = os.path.join("../data", "LiverImages")
    slide_list = get_slide_list(slides_dir)
    check_slide_level(slide_list)
