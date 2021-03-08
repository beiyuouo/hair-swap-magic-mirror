# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/21 21:57
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
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet', n_classes=3),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet', n_classes=3),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18', n_classes=3),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34', n_classes=3),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50', n_classes=3),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101', n_classes=3),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152', n_classes=3)
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


def train():
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
    traindata = HeadSegData(args.seg_data_path, args.train_txt, train=True)
    train_loader = DataLoader(traindata, batch_size=args.seg_batch_size, shuffle=True, num_workers=1)

    net, _ = build_network(None, args.seg_backend)
    seg_criterion = nn.NLLLoss().cuda(0)
    cls_criterion = nn.BCEWithLogitsLoss().cuda(0)
    optimizer = optim.Adam(net.parameters(), lr=args.seg_lr)
    # scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])

    print("start training...")
    net.train()
    total_loss = 0.0
    for epoch in range(args.seg_epochs):
        if (epoch+1) % 5 == 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.25
        total_loss = 0.0
        for i, (x, y, y_cls) in enumerate(train_loader):
            x, y, y_cls = x.cuda(0), y.cuda(0).long(), y_cls.cuda(0).float()

            out, out_cls = net(x)
            # print(x.shape, out.shape, out_cls.shape, y.shape, y_cls.shape)
            seg_loss = seg_criterion(out, y)
            cls_loss = cls_criterion(out_cls, y_cls)
            loss = seg_loss + args.seg_alpha * cls_loss
            total_loss += loss.item()

            if i % 50 == 0:
                status = '[batch:{0}/{1} epoch:{2}] loss = {3:0.5f}'.format(i, len(traindata) // args.seg_batch_size,
                                                                            epoch + 1,
                                                                            loss.item())
                print(status)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(args.model_path,
                                                  f'{"seg"}_{args.seg_model}_{args.seg_backend}_{epoch}.pth'))
        print(f'epoch:{epoch} total_loss: {total_loss / len(traindata)}')


if __name__ == '__main__':
    train()
