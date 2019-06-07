# -*- coding: utf-8 -*-

import os, sys
import pandas as pd


def optimize_burden_csv(phase1_path):
    df = pd.read_csv(phase1_path)
    df = df[:20] # keep front 20
    df['wsi_id'] = df['wsi_id'].apply(lambda x: int(x[:-3]))
    save_path = os.path.join(os.path.dirname(phase1_path), "Phase_1_tumor_burden.csv")
    df.to_csv(save_path)


if __name__ == "__main__":
    # put all download zipped files here
    source_slides_dir = "../data/SourceData"
    phase1_path = os.path.join(source_slides_dir, "Phase_1_tumor_burdenBK.csv")
    optimize_burden_csv(phase1_path)
