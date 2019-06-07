# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io, transform
import scipy.misc as misc
import matplotlib.pyplot as plt

from pydaily import filesystem
from pyslide import pyramid



def get_slide_filenames(slides_dir):
    slide_list = []
    svs_file_list = filesystem.find_ext_files(slides_dir, "svs")
    slide_list.extend([os.path.basename(ele) for ele in svs_file_list])
    SVS_file_list = filesystem.find_ext_files(slides_dir, "SVS")
    slide_list.extend([os.path.basename(ele) for ele in SVS_file_list])
    slide_filenames = [os.path.splitext(ele)[0] for ele in slide_list]

    return slide_filenames


def save_mask_compare(slides_dir, slide_filenames):
    slide_num = len(slide_filenames)
    mask_save_dir = os.path.join(os.path.dirname(slides_dir), "Visualization/Masks")
    filesystem.overwrite_dir(mask_save_dir)
    for ind in np.arange(slide_num):
        print("processing {}/{}".format(ind+1, slide_num))
        check_slide_mask(slides_dir, slide_filenames, ind)


def check_slide_mask(slides_dir, slide_filenames, slide_index, display_level=2):
    """ Load slide segmentation mask.

    """

    slide_path = os.path.join(slides_dir, slide_filenames[slide_index]+".svs")
    if not os.path.exists(slide_path):
        slide_path = os.path.join(slides_dir, slide_filenames[slide_index]+".SVS")
    wsi_head = pyramid.load_wsi_head(slide_path)
    new_size = (wsi_head.level_dimensions[display_level][1], wsi_head.level_dimensions[display_level][0])
    slide_img = wsi_head.read_region((0, 0), display_level, wsi_head.level_dimensions[display_level])
    slide_img = np.asarray(slide_img)[:,:,:3]

    # load and resize whole mask
    whole_mask_path = os.path.join(slides_dir, slide_filenames[slide_index]+"_whole.tif")
    whole_mask_img = io.imread(whole_mask_path)
    resize_whole_mask = (transform.resize(whole_mask_img, new_size, order=0) * 255).astype(np.uint8)
    # load and resize viable mask
    viable_mask_path = os.path.join(slides_dir, slide_filenames[slide_index]+"_viable.tif")
    viable_mask_img = io.imread(viable_mask_path)
    resize_viable_mask = (transform.resize(viable_mask_img, new_size, order=0) * 255).astype(np.uint8)

    # show the mask
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    ax1.imshow(slide_img)
    ax1.set_title('Slide Image')
    ax2.imshow(resize_whole_mask)
    ax2.set_title('Whole Tumor Mask')
    ax3.imshow(resize_viable_mask)
    ax3.set_title('Viable Tumor Mask')
    plt.tight_layout()
    # plt.show()
    save_path = os.path.join(os.path.dirname(slides_dir), "Visualization/Masks", slide_filenames[slide_index]+".png")
    fig.savefig(save_path)

if __name__ == "__main__":
    slides_dir = os.path.join("../data", "LiverImages")
    slide_filenames = get_slide_filenames(slides_dir)
    # check_slide_mask(slides_dir, slide_filenames, 28)
    save_mask_compare(slides_dir, slide_filenames)
