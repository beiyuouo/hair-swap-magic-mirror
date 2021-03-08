# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/3/7 21:47
# Description:

__author__ = "BeiYu"

import os

import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    path = os.path.join('seg_data', 'data')
    train_dataset = ImageFolder(root=path, transform=transforms.ToTensor())
    print(getStat(train_dataset))

