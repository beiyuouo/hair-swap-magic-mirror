# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/3/8 00:55
# Description:

__author__ = "BeiYu"

import numpy as np


class LabelProcessor:

    def __init__(self):
        self.colormap = [(0, 0, 0),
                         (128, 128, 128),
                         (255, 255, 255)]

        self.color2label = self.encode_label_pix(self.colormap)

    @staticmethod
    def get_dis(x, y):
        xx, xy, xz = x // 256 // 256, x // 256 % 256, x % 256
        yx, yy, yz = y // 256 // 256, y // 256 % 256, y % 256

        return (xx-yx)**2 + (xy-yy)**2 + (xz-yz)**2

    def encode_label_pix(self, colormap, cached=False):
        if cached:
            cm21b = np.load('cm21b.npy')
            return cm21b
        cm2lb = np.zeros(256 ** 3)
        for i in range(cm2lb.shape[0]):
            dis = 3*(256**2)
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
