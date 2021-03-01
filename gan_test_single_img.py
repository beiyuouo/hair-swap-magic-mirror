# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/3/1 22:00
# Description:

__author__ = "BeiYu"

import torch.utils.data
from scipy import misc
from torch import optim
from torchvision.utils import save_image

from modules.gan import GeneratorResNet, Discriminator
from modules.vae import *
import numpy as np
import pickle
import time
import random
import os

from modules.gan_dataset import HeadGanData
from utils.init_env import set_seed
from utils.options import *
from torch.utils.data import DataLoader

from PIL import Image


def train():
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    traindata = HeadGanData(args.gan_data_path, args.trainxml)
    train_loader = DataLoader(traindata, batch_size=args.gan_batch_size, shuffle=True, num_workers=1)

    # print(traindata[0])

    f_vae = VAE(zsize=args.vae_zsize, layer_count=5)
    f_vae.load_state_dict(torch.load(os.path.join(args.model_path, 'f_vae_256_19.pth')))
    f_vae.cuda()
    f_vae.eval()

    h_vae = VAE(zsize=args.vae_zsize, layer_count=5)
    h_vae.load_state_dict(torch.load(os.path.join(args.model_path, 'h_vae_256_19.pth')))
    h_vae.cuda()
    h_vae.eval()

    G = GeneratorResNet(zsize=args.vae_zsize, img_shape=(3, 128, 128))
    G.load_state_dict(torch.load(os.path.join(args.model_path, 'G_19.pth')))
    D = Discriminator(img_shape=(3, 128, 128))
    D.load_state_dict(torch.load(os.path.join(args.model_path, 'D_19.pth')))
    G.cuda()
    D.cuda()
    G.eval()
    D.eval()

    print('Start testing...')

    fake_img = None

    for i, (f_img, h_img, a_img) in enumerate(train_loader):
        f_img, h_img, a_img = f_img.cuda(), h_img.cuda(), a_img.cuda()
        z_f, z_h = f_vae.get_z(f_img), h_vae.get_z(h_img)

        minibs = f_img.size(0)

        fake_img = G(z_f, z_h)

        for idx in range(minibs):
            img = traindata.inv_transform(fake_img[idx].view(1, 3, 128, 128)).convert('RGB')
            # print(img.shape)
            img.save(os.path.join('result', f'{i}_{idx}.jpg'))
            # img.show()

        if i > 10:
            break


if __name__ == '__main__':
    train()
