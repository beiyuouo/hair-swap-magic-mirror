# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/22 18:49
# Description:

__author__ = "BeiYu"

import os
import os.path as osp
from copy import deepcopy

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


class HeadVaeData(data.Dataset):
    def __init__(self, datadir, train_txt='segmentations.txt', crop_size=(218, 178), train=True):
        self.datadir = datadir
        self.crop_size = crop_size
        self.train = train
        self.train_txt = train_txt
        self.src = []
        self.mask = []
        self.img_dir = os.path.join(datadir, 'data')
        self.mask_dir = os.path.join(datadir, 'segmentation_masks')
        self.rgb_mean = np.array([0.5320177, 0.44282365, 0.39255527])
        self.rgb_std = np.array([0.25269014, 0.23634945, 0.23565148])

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

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # img, mask = self.center_crop(img, mask, self.crop_size)

        img, img_f, img_h = self.img_transform(img, mask, index)
        # print(mask.shape)

        return img, img_f, img_h

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

    def inv_transform(self, img_tensor):
        inv_normalize = torchvision.transforms.Normalize(
            mean=-self.rgb_mean / self.rgb_std,
            std=1 / self.rgb_std)
        to_PIL_image = torchvision.transforms.ToPILImage()
        return to_PIL_image(inv_normalize(img_tensor[0].cpu()).clamp(0, 1))

    def center_crop(self, img, mask, crop_size):
        img = ff.center_crop(img, crop_size)
        mask = ff.center_crop(mask, crop_size)

        return img, mask

    def overlay_mask_with_color(self, img, seg_mask, color):
        color_img = np.zeros(img.shape, img.dtype)
        color_img[:, :] = img[:, :]

        color_mask = cv2.bitwise_and(color_img, color_img, mask=seg_mask)
        # display_image = cv2.addWeighted(color_mask, 0.3, img, 0.7, 0)
        return color_mask

    def img_transform(self, img, mask, index):
        transform_img = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5320177, 0.44282365, 0.39255527], std=[0.25269014, 0.23634945, 0.23565148])
            ]
        )

        hair_mask = np.zeros(mask.shape, dtype=np.uint8)
        hair_mask[mask == g_label_hair] = 0
        img_h = self.overlay_mask_with_color(img, hair_mask, g_hair_color_display)

        skin_mask = np.zeros(mask.shape, dtype=np.uint8)
        skin_mask[mask == g_label_face] = 0
        img_f = self.overlay_mask_with_color(img, skin_mask, g_skin_color_display)

        img, img_f, img_h = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), \
                            Image.fromarray(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)), \
                            Image.fromarray(cv2.cvtColor(img_h, cv2.COLOR_BGR2RGB))

        # img.show()
        # img_f.show()
        # img_h.show()

        img, img_f, img_h = transform_img(img), transform_img(img_f), transform_img(img_h)

        return img, img_f, img_h
