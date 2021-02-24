# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/22 09:46
# Description:

__author__ = "BeiYu"

from utils.init_env import set_seed
from utils.options import *

import os
import logging
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from modules.seg_dataset import *
from tqdm import tqdm
import click
import torch.nn.functional as F
import numpy as np
from modules.seg import PSPNet

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    # net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda(0)
    return net, epoch


def test():
    args = get_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # net, starting_epoch = build_network(snapshot, backend)
    # data_path = os.path.abspath(os.path.expanduser(data_path))
    # models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(args.model_path, exist_ok=True)
    set_seed(args.seed)

    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    testdata = HeadSegData(args.seg_data_path, args.trainxml, train=False, crop_size=(args.seg_crop_x, args.seg_crop_y))
    test_loader = DataLoader(testdata, batch_size=args.seg_batch_size, shuffle=True, num_workers=1)

    net, _ = build_network(None, args.seg_backend)
    net.load_state_dict(torch.load(os.path.join(args.model_path, 'seg_PSPNet_resnet34_31.pth')))
    seg_criterion = nn.NLLLoss().cuda(0)
    cls_criterion = nn.BCEWithLogitsLoss().cuda(0)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])

    print("start testing...")
    net.eval()

    total_loss = 0.0

    with torch.no_grad():
        for i, (x, y, y_cls) in enumerate(test_loader):
            x, y, y_cls = x.cuda(0), y.cuda(0).long(), y_cls.cuda(0).float()
            # print(x.shape)

            out, out_cls = net(x)
            seg_loss = seg_criterion(out, y)
            cls_loss = cls_criterion(out_cls, y_cls)
            loss = seg_loss + args.seg_alpha * cls_loss
            total_loss += loss.item()

            # optimizer.step()
            if (i+1) % 50 == 0:
                print(f'test: batch[{i}/{len(testdata)}] loss: {loss.item() / len(x)}')

    print(f'total loss: {total_loss / len(testdata)}')


if __name__ == '__main__':
    test()

