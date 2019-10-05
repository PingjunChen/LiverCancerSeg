# -*- coding: utf-8 -*-

import os, sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import argparse, uuid, time
from skimage import io, transform
from tifffile import imsave
from collections import defaultdict
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from pydaily import filesystem
from pyslide import patch, pyramid

from segnet import UNet, pspnet
from utils import get_slide_filenames, gen_patch_wmap
from utils import wsi_stride_splitting
from loss import calc_loss
from dataload import PatchDataset


def set_args():
    parser = argparse.ArgumentParser(description = 'Liver Tumor Slide Segmentation')
    parser.add_argument("--class_num",       type=int,   default=1)
    parser.add_argument("--in_channels",     type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=48)
    parser.add_argument("--stride_len",      type=int,   default=128)
    parser.add_argument("--patch_len",       type=int,   default=512)
    parser.add_argument("--slide_level",     type=int,   default=2)
    parser.add_argument("--model_name",      type=str,   default="PSP")
    parser.add_argument("--gpu",             type=str,   default="3,5,6,7")
    parser.add_argument("--split",           type=str,   default="BestModel")
    parser.add_argument("--tumor_type",      type=str,   default="viable")
    parser.add_argument("--best_model",      type=str,   default="PSP-050-0.755.pth")
    # parser.add_argument("--tumor_type",      type=str,   default="whole")
    # parser.add_argument("--best_model",      type=str,   default="PSP-049-0.682.pth")

    parser.add_argument("--model_dir",       type=str,   default="../data/Models")
    parser.add_argument("--slides_dir",      type=str,   default="../data/TestSlides")
    parser.add_argument("--result_dir",      type=str,   default="../data/TestResults")
    parser.add_argument("--normalize",       type=bool,  default=False)
    parser.add_argument("--save_org",        type=bool,  default=True)
    parser.add_argument("--seed",            type=int,   default=1234)

    args = parser.parse_args()
    return args


def test_slide_seg(args):
    model = None
    if args.model_name == "UNet":
        model = UNet(n_channels=args.in_channels, n_classes=args.class_num)
    elif args.model_name == "PSP":
        model = pspnet.PSPNet(n_classes=19, input_size=(512, 512))
        model.classification = nn.Conv2d(512, args.class_num, kernel_size=1)
    else:
        raise AssertionError("Unknow modle: {}".format(args.model_name))
    model_path = os.path.join(args.model_dir, args.tumor_type, args.split, args.best_model)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    since = time.time()
    result_dir = os.path.join(args.result_dir, args.tumor_type)
    filesystem.overwrite_dir(result_dir)
    slide_names = get_slide_filenames(args.slides_dir)
    if args.save_org and args.tumor_type == "viable":
        org_result_dir = os.path.join(result_dir, "Level0")
        filesystem.overwrite_dir(org_result_dir)

    for num, cur_slide in enumerate(slide_names):
        print("--{:02d}/{:02d} Slide:{}".format(num+1, len(slide_names), cur_slide))
        metrics = defaultdict(float)
        # load level-2 slide
        slide_path = os.path.join(args.slides_dir, cur_slide+".svs")
        if not os.path.exists(slide_path):
            slide_path = os.path.join(args.slides_dir, cur_slide+".SVS")
        wsi_head = pyramid.load_wsi_head(slide_path)
        p_level = args.slide_level
        pred_h, pred_w = (wsi_head.level_dimensions[p_level][1], wsi_head.level_dimensions[p_level][0])
        slide_img = wsi_head.read_region((0, 0), p_level, wsi_head.level_dimensions[p_level])
        slide_img = np.asarray(slide_img)[:,:,:3]

        coors_arr = wsi_stride_splitting(pred_h, pred_w, patch_len=args.patch_len, stride_len=args.stride_len)
        patch_arr, wmap = gen_patch_wmap(slide_img, coors_arr, plen=args.patch_len)
        patch_dset = PatchDataset(patch_arr, mask_arr=None, normalize=args.normalize, tumor_type=args.tumor_type)
        patch_loader = DataLoader(patch_dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)
        ttl_samples = 0
        pred_map = np.zeros_like(wmap).astype(np.float32)
        for ind, patches in enumerate(patch_loader):
            inputs = Variable(patches.cuda())
            with torch.no_grad():
                outputs = model(inputs)
                preds = F.sigmoid(outputs)
                preds = torch.squeeze(preds, dim=1).data.cpu().numpy()
                if (ind+1)*args.batch_size <= len(coors_arr):
                    patch_coors = coors_arr[ind*args.batch_size:(ind+1)*args.batch_size]
                else:
                    patch_coors = coors_arr[ind*args.batch_size:]
                for ind, coor in enumerate(patch_coors):
                    ph, pw = coor[0], coor[1]
                    pred_map[ph:ph+args.patch_len, pw:pw+args.patch_len] += preds[ind]
                ttl_samples += inputs.size(0)

        prob_pred = np.divide(pred_map, wmap)
        slide_pred = (prob_pred > 0.5).astype(np.uint8)
        pred_save_path = os.path.join(result_dir, cur_slide + "_" + args.tumor_type + ".tif")
        io.imsave(pred_save_path, slide_pred*255)

        if args.save_org and args.tumor_type == "viable":
            org_w, org_h = wsi_head.level_dimensions[0]
            org_pred = transform.resize(prob_pred, (org_h, org_w))
            org_pred = (org_pred > 0.5).astype(np.uint8)
            org_save_path = os.path.join(org_result_dir, cur_slide[-3:] + ".tif")
            imsave(org_save_path, org_pred, compress=9)

    time_elapsed = time.time() - since
    print('Testing takes {:.0f}m {:.2f}s'.format(time_elapsed // 60, time_elapsed % 60))

if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # train model
    print("{} prediction using: {}, model: {}".format(args.tumor_type.upper(), args.model_name, args.best_model))
    test_slide_seg(args)
