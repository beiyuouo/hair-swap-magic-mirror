# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/22 10:04
# Description:

__author__ = "BeiYu"

from utils.options import *

import os
import logging
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from modules.dataset import *
from tqdm import tqdm
import click
import torch.nn.functional as F
import numpy as np
from modules.segmentation import PSPNet
import torchvision.transforms.functional as ff


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


def get_img(data_path, path, save_path=None):
    img_file = path
    img_path = os.path.join(data_path, img_file).replace("\\", "/")
    img = Image.open(img_path).convert("RGB")
    crop_size = (296, 280)
    img = ff.center_crop(img, crop_size)
    if save_path:
        img.save(save_path)
    img.show()
    transform_img = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    img = transform_img(img)
    return img


colormap = [[0, 0, 0],[255, 0, 0],[0, 255, 0],[255, 255, 0],[128, 128, 128],[0, 0, 255],[255, 0, 255],[0, 255, 255],[255, 255, 255]]

cm = np.array(colormap).astype('uint8')


def recover_img(out, result_path):
    pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
    pre_label = np.asarray(pre_label, dtype=np.uint8)
    pre = cm[pre_label]
    pre_img = Image.fromarray(pre.astype("uint8"), mode='RGB')
    pre_img.save(result_path)
    pre_img.show()


def test():
    args = get_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # net, starting_epoch = build_network(snapshot, backend)
    # data_path = os.path.abspath(os.path.expanduser(data_path))
    # models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(args.model_path, exist_ok=True)

    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''

    net, _ = build_network(None, args.backend)
    net.load_state_dict(torch.load(os.path.join(args.model_path, '19.pth')))
    # scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])

    print("start testing...")
    net.eval()

    img_name = '10696954214_ee49a7428f_o_scaled_scaled.jpg'
    img_name = '10933518394_003d1613a7_o_scaled.jpg'
    path = os.path.join('real_photos', img_name)
    x = get_img(args.data_path, path, os.path.join(args.result_path, img_name))
    x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
    x = x.cuda(0)
    out, out_cls = net(x)
    # print(out)
    recover_img(out, os.path.join(args.result_path, 'pred_'+img_name))
    print(out_cls.cpu().data.numpy())


if __name__ == '__main__':
    test()
