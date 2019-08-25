# -*- coding: utf-8 -*-

import os, sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import argparse
import time, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.backends.cudnn as cudnn

from collections import defaultdict
from segnet import UNet, pspnet

from dataload import gen_dloader
from loss import calc_loss, print_metrics
from utils import LambdaLR


def set_args():
    parser = argparse.ArgumentParser(description = 'Liver Tumor Patch Segmentation')
    parser.add_argument("--class_num",       type=int,   default=1)
    parser.add_argument("--batch_size",      type=int,   default=8,        help="batch size")
    parser.add_argument("--in_channels",     type=int,   default=3,        help="input channel number")
    parser.add_argument("--maxepoch",        type=int,   default=100,      help="number of epochs to train")
    parser.add_argument("--init_lr",         type=float, default=1.0e-2,   help="init learning rate for optimization")
    # parser.add_argument("--decay_epoch",     type=int,   default=12,       help="lr start to decay linearly from decay_epoch")
    parser.add_argument("--bce_weight",      type=float, default=0.1,      help="weight of bce loss")
    parser.add_argument("--data_dir",        type=str,   default="../data/Patches")
    parser.add_argument("--model_dir",       type=str,   default="../data/Models")
    parser.add_argument("--tumor_type",      type=str,   default="viable")
    parser.add_argument("--normalize",       type=bool,  default=False)
    parser.add_argument("--model_name",      type=str,   default="PSP")
    parser.add_argument("--optim_name",      type=str,   default="SGD")
    parser.add_argument("--gpu",             type=str,   default="2, 3",   help="training gpu")
    parser.add_argument("--seed",            type=int,   default=1234,     help="training seed")
    parser.add_argument("--session",         type=str,   default="LR01",   help="training session")

    args = parser.parse_args()
    return args


def train_seg_model(args):
    # model
    model = None
    if args.model_name == "UNet":
        model = UNet(n_channels=args.in_channels, n_classes=args.class_num)
    elif args.model_name == "PSP":
        model = pspnet.PSPNet(n_classes=19, input_size=(512, 512))
        model.load_pretrained_model(model_path="./segnet/pspnet/pspnet101_cityscapes.caffemodel")
        model.classification = nn.Conv2d(512, args.class_num, kernel_size=1)
    else:
        raise AssertionError("Unknow modle: {}".format(args.model_name))
    model = nn.DataParallel(model)
    model.cuda()
    # optimizer
    optimizer = None
    if args.optim_name == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1.0e-3)
    elif args.optim_name == "SGD":
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=args.init_lr, momentum=0.9, weight_decay=0.0005)
    else:
        raise AssertionError("Unknow optimizer: {}".format(args.optim_name))
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(args.maxepoch, 0, 0).step)
    # dataloader
    train_data_dir = os.path.join(args.data_dir, args.tumor_type, "train")
    train_dloader = gen_dloader(train_data_dir, args.batch_size,  mode="train", normalize=args.normalize, tumor_type=args.tumor_type)
    test_data_dir = os.path.join(args.data_dir, args.tumor_type, "val")
    val_dloader = gen_dloader(test_data_dir, args.batch_size,  mode="val", normalize=args.normalize, tumor_type=args.tumor_type)

    # training
    save_model_dir = os.path.join(args.model_dir, args.tumor_type, args.session)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    best_dice = 0.0
    for epoch in np.arange(0, args.maxepoch):
        print('Epoch {}/{}'.format(epoch+1, args.maxepoch))
        print('-' * 10)
        since = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                dloader = train_dloader
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("Current LR: {:.8f}".format(param_group['lr']))
                model.train()  # Set model to training mode
            else:
                dloader = val_dloader
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            for batch_ind, (imgs, masks) in enumerate(dloader):
                inputs = Variable(imgs.cuda())
                masks = Variable(masks.cuda())
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, masks, metrics, bce_weight=args.bce_weight)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                epoch_samples += inputs.size(0)
            print_metrics(metrics, epoch_samples, phase)
            epoch_dice = metrics['dice'] / epoch_samples

            # deep copy the model
            if phase == 'val' and (epoch_dice > best_dice or epoch > args.maxepoch-5):
                best_dice = epoch_dice
                best_model = copy.deepcopy(model.state_dict())
                best_model_name = "-".join([args.model_name, "{:03d}-{:.3f}.pth".format(epoch, best_dice)])
                torch.save(best_model, os.path.join(save_model_dir, best_model_name))
        time_elapsed = time.time() - since
        print('Epoch {:2d} takes {:.0f}m {:.0f}s'.format(epoch, time_elapsed // 60, time_elapsed % 60))
    print("================================================================================")
    print("Training finished...")


if  __name__ == '__main__':
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    # train model
    print("Training {} with {} on {}".format(args.model_name, args.optim_name, args.tumor_type))
    train_seg_model(args)
