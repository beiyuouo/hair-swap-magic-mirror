# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/22 23:42
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


colormap = [[0, 0, 0], [128, 128, 128], [255, 255, 255]]

cm = np.array(colormap).astype('uint8')


def save_x_img(x, result_path):
    x = x[0].cpu().data.numpy().astype("uint8")
    x = x.swapaxes(0, 1).swapaxes(1, 2)
    print(x.shape)
    img = Image.fromarray(x, mode='RGB')
    img.save(result_path)


def recover_img(out, result_path):
    pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
    pre_label = np.asarray(pre_label, dtype=np.uint8)
    pre = cm[pre_label]
    pre_img = Image.fromarray(pre.astype("uint8"), mode='RGB')
    pre_img.save(result_path)
    # pre_img.show()


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
    testdata = HeadSegData(args.seg_data_path, args.train_txt, train=False)
    test_loader = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=1)

    net, _ = build_network(None, args.seg_backend)
    net.load_state_dict(torch.load(os.path.join(args.model_path, 'seg_PSPNet_resnet34_49.pth')))
    seg_criterion = nn.NLLLoss().cuda(0)
    cls_criterion = nn.BCEWithLogitsLoss().cuda(0)

    print("start running...")
    net.eval()

    total_loss = 0.0

    with torch.no_grad():
        for i, (x, y, y_cls) in enumerate(test_loader):
            x, y, y_cls = x.cuda(0), y.cuda(0).long(), y_cls.cuda(0).float()
            testdata.save_to(i, os.path.join(args.result_path, f'real_{str(i)}.jpg'))
            # print(x.shape)

            out, out_cls = net(x)
            seg_loss = seg_criterion(out, y)
            cls_loss = cls_criterion(out_cls, y_cls)
            loss = seg_loss + args.seg_alpha * cls_loss
            total_loss += loss.item()

            # optimizer.step()
            if (i + 1) % 50 == 0:
                print(f'test: batch[{i}/{len(testdata)}] loss: {loss.item() / len(x)}')

            recover_img(out, os.path.join(args.result_path, f'pred_{str(i)}.jpg'))

    print(f'total loss: {total_loss / len(testdata)}')


if __name__ == '__main__':
    test()
