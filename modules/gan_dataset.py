# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/22 18:49
# Description:

__author__ = "BeiYu"

import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import torchvision.transforms.functional as ff
from modules.seg_augmentation import *
from torchvision import transforms
import torch

import xml.etree.ElementTree as ET


class LabelProcessor:

    def __init__(self):
        self.colormap = [(0, 0, 0),
                         (255, 0, 0),
                         (0, 255, 0),
                         (255, 255, 0),
                         (128, 128, 128),
                         (0, 0, 255),
                         (255, 0, 255),
                         (0, 255, 255),
                         (255, 255, 255)]

        self.color2label = self.encode_label_pix(self.colormap)

    @staticmethod
    def get_dis(x, y):
        xx, xy, xz = x // 256 // 256, x // 256 % 256, x % 256
        yx, yy, yz = y // 256 // 256, y // 256 % 256, y % 256

        return (xx - yx) ** 2 + (xy - yy) ** 2 + (xz - yz) ** 2

    def encode_label_pix(self, colormap, cached=True):
        if cached:
            cm21b = np.load('cm21b.npy')
            return cm21b
        cm2lb = np.zeros(256 ** 3)
        for i in range(cm2lb.shape[0]):
            dis = 3 * (256 ** 2)
            idx = 0
            for ii, cm in enumerate(colormap):
                if dis > self.get_dis(i, (cm[0] * 256 + cm[1]) * 256 + cm[2]):
                    dis = self.get_dis(i, (cm[0] * 256 + cm[1]) * 256 + cm[2])
                    idx = ii
            cm2lb[i] = idx
        np.save('cm21b.npy', cm2lb)
        return cm2lb

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        label = np.array(self.color2label[idx], dtype='int64')

        return label


p = LabelProcessor()


class HeadGanData(data.Dataset):
    def __init__(self, datadir, train=True):
        self.datadir = datadir
        self.train = train

        # self.p = LabelProcessor()

    def __getitem__(self, index):
        img_file = f'real_{index}.jpg'
        label_file = f'pred_{index}.jpg'

        img_path = os.path.join(self.datadir, img_file).replace("\\", "/")
        label_path = os.path.join(self.datadir, label_file).replace("\\", "/")

        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        f_img, h_img, a_img = self.img_transform(img, label, index)

        return f_img, h_img, a_img

    def __len__(self):
        return len(os.listdir(self.datadir)) // 2

    def img_transform(self, img, label, index):
        label = np.array(label)

        label = p.encode_label_img(label)

        mask_h = label[:, :] != 3
        mask_h = np.expand_dims(mask_h, 2).repeat(3, axis=2)

        mask_f = (label[:, :] == 3) | (label[:, :] == 0)
        mask_f = np.expand_dims(mask_f, 2).repeat(3, axis=2)

        mask_a = label[:, :] == 0
        mask_a = np.expand_dims(mask_a, 2).repeat(3, axis=2)
        # print(mask_f.shape)

        # for i in range(label.shape[0]):
        #    for j in range(label.shape[1]):
        #        print(label[i][j], end=' ')
        #    print('')
        # print(label.shape)

        f_img, h_img, a_img = np.array(img), np.array(img), np.array(img)
        f_img[mask_f] = 0
        h_img[mask_h] = 0
        a_img[mask_a] = 0
        # f_img = np.ma.array(img, mask=mask_f)
        # h_img = np.ma.array(img, mask=mask_h)

        transform_label = transforms.Compose([
            transforms.ToTensor()]
        )

        transform_img = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ])

        # Image.fromarray(f_img).show()
        # Image.fromarray(h_img).show()
        # Image.fromarray(a_img).show()

        f_img = transform_img(Image.fromarray(f_img))
        h_img = transform_img(Image.fromarray(h_img))
        a_img = transform_img(Image.fromarray(a_img))

        return f_img, h_img, a_img
