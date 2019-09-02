# -*- coding: utf-8 -*-

import os, sys
import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
from skimage import io
from pydaily import format

def set_args():
    parser = argparse.ArgumentParser(description = 'Liver Burden Prediction')

    parser.add_argument("--model_name",      type=str,   default="PSP")
    parser.add_argument("--result_dir",      type=str,   default="../data/TestResults")
    parser.add_argument("--seed",            type=int,   default=1234)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    viable_pred_dir = os.path.join(args.result_dir, "viable")
    whole_pred_dir = os.path.join(args.result_dir, "whole")
    if not os.path.exists(viable_pred_dir):
        raise Exception("No viable tumor prediction yet")
    if not os.path.exists(whole_pred_dir):
        raise Exception("No whole tumor prediciton yet")

    viable_pred_list = [ele for ele in os.listdir(viable_pred_dir) if "viable" in ele]
    whole_pred_list = [ele for ele in os.listdir(whole_pred_dir) if "whole" in ele]
    if len(viable_pred_list) != len(whole_pred_list):
        print("The number of viable and whole is not equal.")


    slide_list, ratio_list = [], []
    id_list = [ele[7:10] for ele in viable_pred_list]
    id_list.sort()
    for id in id_list:
        viable_pred_name = "01_01_0" + id + "_viable.tif"
        whole_pred_name = "01_01_0" + id + "_whole.tif"
        viable_pred_path = os.path.join(viable_pred_dir, viable_pred_name)
        whole_pred_path = os.path.join(whole_pred_dir, whole_pred_name)
        viable_pred_img = io.imread(viable_pred_path) / 255
        whole_pred_img = io.imread(whole_pred_path) / 255
        pred_burden = np.sum(viable_pred_img) * 100.0 / (np.sum(whole_pred_img) + 1.0e-8)
        print("{} {:.3f}".format(id, pred_burden))
        slide_list.append(id)
        pred_burden = min(pred_burden, 99.999)
        ratio_list.append(round(pred_burden, 3))

    # save the prediction for submission
    burden_dict = {}
    burden_dict["wsi_id"] = slide_list
    burden_dict["ratio"] = ratio_list
    pred_csv_path = os.path.join(args.result_dir, "prediction.csv")
    format.dict_to_csv(burden_dict, pred_csv_path)
