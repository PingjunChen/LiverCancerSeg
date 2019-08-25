# -*- coding: utf-8 -*-

import os, sys
import numpy as np
from skimage import io
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


viable_rgb_mean = (0.790, 0.614, 0.739)
viable_rgb_std = (0.093, 0.128, 0.102)
whole_rgb_mean = (0.765, 0.547, 0.692)
whole_rgb_std = (0.092, 0.119, 0.098)


class LiverPatchDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.patch_dir = os.path.join(self.data_dir, "imgs")
        self.mask_dir = os.path.join(self.data_dir, "masks")
        self.patch_list = [ele for ele in os.listdir(self.patch_dir) if "jpg" in ele]
        self.transform = transform
        self.cur_img_name = None


    def __len__(self):
        return len(self.patch_list)


    def __getitem__(self, idx):
        self.cur_img_name = os.path.splitext(self.patch_list[idx])[0]
        patch_path = os.path.join(self.patch_dir, self.patch_list[idx])
        mask_path = os.path.join(self.mask_dir, self.cur_img_name + ".png")

        image = (io.imread(patch_path) / 255.0).astype(np.float32)
        mask = (io.imread(mask_path) / 255.0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        if self.transform:
            image = self.transform(image)

        return [image, mask]


def gen_dloader(data_dir, batch_size, mode="train", normalize=True, tumor_type="viable"):
    transform_list = [transforms.ToTensor(), ]
    if tumor_type == "viable":
        rgb_mean, rgb_std = viable_rgb_mean, viable_rgb_std
    elif tumor_type == "whole":
        rgb_mean, rgb_std = whole_rgb_mean, whole_rgb_std
    else:
        raise AssertionError("Unknow tumor type: {}".format(tumor_type))
    if normalize == True:
        transform_list.append(transforms.Normalize(rgb_mean, rgb_std))
    trans = transforms.Compose(transform_list)


    dset = LiverPatchDataset(data_dir, transform=trans)
    if mode == "train":
        dloader = DataLoader(dset, batch_size=batch_size, shuffle=True,
                             num_workers=4, drop_last=True)
    elif mode == "val":
        dloader = DataLoader(dset, batch_size=batch_size, shuffle=False,
                             num_workers=4, drop_last=False)
    else:
        raise Exception("Unknow mode: {}".format(mode))


    return dloader



class PatchDataset(Dataset):
    """
    Dataset for slide testing. Each would be splitted into multiple patches.
    Prediction is made on these splitted patches.
    """

    def __init__(self, patch_arr, mask_arr=None, normalize=True, tumor_type="viable"):
        self.patches = patch_arr
        self.masks = mask_arr
        transform_list = [transforms.ToTensor(), ]
        if tumor_type == "viable":
            rgb_mean, rgb_std = viable_rgb_mean, viable_rgb_std
        elif tumor_type == "whole":
            rgb_mean, rgb_std = whole_rgb_mean, whole_rgb_std
        else:
            raise AssertionError("Unknow tumor type: {}".format(tumor_type))
        if normalize == True:
            transform_list.append(transforms.Normalize(rgb_mean, rgb_std))
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        patch = self.patches[idx,...]
        if self.transform:
            patch = self.transform(patch)
        if isinstance(self.masks, np.ndarray):
            mask = np.expand_dims(self.masks[idx,...], axis=0)
            return patch, mask
        else:
            return patch
