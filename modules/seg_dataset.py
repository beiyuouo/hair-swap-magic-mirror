# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/21 22:37
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

g_label_face = 128
g_label_hair = 255

g_hair_color_display = (0, 0, 255)
g_skin_color_display = (0, 255, 0)


class LabelProcessor:

    def __init__(self):
        self.colormap = [(0, 0, 0),
                         (128, 128, 128),
                         (255, 255, 255)]

        self.color2label = self.encode_label_pix(self.colormap)

    @staticmethod
    def encode_label_pix(colormap):
        cm2lb = np.zeros(256 ** 3)
        for i, cm in enumerate(colormap):
            cm2lb[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

        return cm2lb

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        label = np.array(self.color2label[idx], dtype='int64')

        return label


p = LabelProcessor()


class HeadSegData(data.Dataset):
    def __init__(self, datadir, train_txt='segmentations.txt', crop_size=(218, 178), train=True):
        self.datadir = datadir
        self.crop_size = crop_size
        self.train = train
        self.train_txt = train_txt
        self.src = []
        self.mask = []
        self.img_dir = os.path.join(datadir, 'data')
        self.mask_dir = os.path.join(datadir, 'segmentation_masks')

        with open(self.train_txt, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n')
                self.src.append(line)
                self.mask.append(line.replace('.jpg', '.bmp'))

    def __getitem__(self, index):
        img_file = self.src[index]
        mask_file = self.mask[index]

        img_path = os.path.join(self.img_dir, img_file).replace("\\", "/")
        mask_path = os.path.join(self.mask_dir, mask_file).replace("\\", "/")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        # img, mask = self.center_crop(img, mask, self.crop_size)

        img, mask, y_cls = self.img_transform(img, mask, index)
        # print(mask.shape)

        return img, mask, y_cls

    def __len__(self):
        return len(self.src)

    def save_to(self, idx, path):
        img_file = self.src[idx]
        img_path = os.path.join(self.img_dir, img_file).replace("\\", "/")
        img = Image.open(img_path).convert("RGB")
        img = transforms.Resize((128, 128))(img)
        # img = ff.center_crop(img, self.crop_size)
        img.save(path)

        return img

    def center_crop(self, img, mask, crop_size):
        img = ff.center_crop(img, crop_size)
        mask = ff.center_crop(mask, crop_size)

        return img, mask

    def img_transform(self, img, mask, index):
        mask = np.array(mask)
        # print(mask.shape)
        mask = Image.fromarray(mask.astype('uint8'))

        transform_label = transforms.Compose([
            transforms.Resize((128, 128)), ]
        )

        mask = transform_label(mask)

        transform_img = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5320177, 0.44282365, 0.39255527], std=[0.25269014, 0.23634945, 0.23565148])
            ]
        )

        img = transform_img(img)

        mask = p.encode_label_img(mask)

        # Image.fromarray(label).save("./results_pic/"+str(index)+".png")
        # print(label.shape)
        y_cls, _ = np.histogram(mask, bins=3, range=(-0.5, 3 - 0.5), )
        y_cls = np.asarray(np.asarray(y_cls, dtype=np.bool), dtype=np.uint8)

        # label = torch.squeeze(label)

        return img, torch.from_numpy(mask), torch.from_numpy(y_cls)
