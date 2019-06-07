# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import pandas as pd
from skimage import io
from pydaily import format


def cal_train_burden(slides_dir):
    slide_list = []
    slide_list.extend([ele[6:-4] for ele in os.listdir(slides_dir) if "svs" in ele])
    slide_list.extend([ele[6:-4] for ele in os.listdir(slides_dir) if "SVS" in ele])
    burden_dict = {}
    for ind, cur_slide in enumerate(slide_list):
        cur_slide = str(cur_slide)
        print("Processing {}/{}".format(ind+1, len(slide_list)))
        cur_whole_path = os.path.join(slides_dir, "01_01_"+cur_slide+"_whole.tif")
        whole_mask = io.imread(cur_whole_path)
        cur_viable_path = os.path.join(slides_dir, "01_01_"+cur_slide+"_viable.tif")
        viable_mask = io.imread(cur_viable_path)
        cur_burden = np.sum(viable_mask) * 1.0 / np.sum(whole_mask)
        burden_dict[cur_slide] = cur_burden
    save_json_path = os.path.join(os.path.dirname(slides_dir), "SourceData", "calculated_tumor_burden.json")
    format.dict_to_json(burden_dict, save_json_path)


def extract_csv_burden(csv_path, case_num):
    df = pd.read_csv(csv_path)
    slide_ids = df['wsi_id'].values.tolist()[:case_num]
    slide_burden = df['pixel ratio'].values.tolist()[:case_num]
    burden_dict = {}
    for id, burden in zip(slide_ids, slide_burden):
        burden_dict[str(id).zfill(4)] = burden

    return burden_dict


if __name__ == "__main__":
    # extract prepared ground truth viable tumor burden
    source_slides_dir = "../data/SourceData"
    phase1_path = os.path.join(source_slides_dir, "Phase_1_tumor_burden.csv")
    phase2_path = os.path.join(source_slides_dir, "Phase_2_tumor_burden.csv")
    gt_burden_dict = {}
    phase1_burden_dict = extract_csv_burden(phase1_path, case_num=20)
    gt_burden_dict.update(phase1_burden_dict)
    phase2_burden_dict = extract_csv_burden(phase2_path, case_num=30)
    gt_burden_dict.update(phase2_burden_dict)

    # get calculate viable tumor burden
    slides_dir = os.path.join(os.path.dirname(source_slides_dir), "LiverImages")
    cal_train_burden(slides_dir)

    # load calcualted burden
    cal_burden_path = os.path.join(source_slides_dir, "calculated_tumor_burden.json")
    cal_burden_dict = format.json_to_dict(cal_burden_path)
    # compare gt & cal
    for ind, key in enumerate(gt_burden_dict):
        if key not in cal_burden_dict:
            print("Error: {}".format(key))
        gt_burden = gt_burden_dict[key]
        cal_burden = cal_burden_dict[key]
        if np.absolute(gt_burden-cal_burden) > 0.001:
            print("{}/{} {} gt:{:.3f}, cal:{:.3f}".format(ind+1, len(gt_burden_dict), key,
                                                  gt_burden, cal_burden))
