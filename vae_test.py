# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/3/8 13:26
# Description:

__author__ = "BeiYu"


import torch.utils.data
from scipy import misc
from torch import optim
from torchvision.utils import save_image
from modules.vae import *
import numpy as np
import pickle
import time
import random
import os

from modules.vae_dataset import HeadVaeData
from utils.init_env import set_seed
from utils.options import *
from torch.utils.data import DataLoader


def loss_function(recon_x, x, mu, logvar):
    BCE = torch.mean((recon_x - x) ** 2)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1


def train():
    args = get_args()
    set_seed(args.seed)

    traindata = HeadVaeData(args.vae_data_path, args.train_txt)
    train_loader = DataLoader(traindata, batch_size=1, shuffle=False, num_workers=1)

    # print(traindata[0])

    f_vae = VAE(zsize=args.vae_zsize, layer_count=5)
    f_vae.cuda()
    f_vae.train()
    f_vae.load_state_dict(torch.load(os.path.join(args.model_path, 'f_vae_256_49.pth')))

    h_vae = VAE(zsize=args.vae_zsize, layer_count=5)
    h_vae.cuda()
    h_vae.load_state_dict(torch.load(os.path.join(args.model_path, 'h_vae_256_49.pth')))

    sample1 = torch.randn(128, args.vae_zsize).view(-1, args.vae_zsize, 1, 1)

    print('Start testing...')

    f_vae.eval()
    h_vae.eval()
    for idx, (x, f_x, h_x) in enumerate(train_loader):
        f_x, h_x = f_x.cuda(), h_x.cuda()

        f_rec, mu, logvar = f_vae(f_x)

        h_rec, mu, logvar = h_vae(h_x)

        minibs = f_x.size(0)
        for i in range(minibs):
            img = traindata.inv_transform(f_x[i].view(1, 3, 128, 128)).convert('RGB')
            # print(img.shape)
            img.save(os.path.join('result', f'{i + idx}_f_real.jpg'))

            img = traindata.inv_transform(h_x[i].view(1, 3, 128, 128)).convert('RGB')
            # print(img.shape)
            img.save(os.path.join('result', f'{i + idx}_h_real.jpg'))

            img = traindata.inv_transform(f_rec[i].view(1, 3, 128, 128)).convert('RGB')
            # print(img.shape)
            img.save(os.path.join('result', f'{i + idx}_f_fake.jpg'))

            img = traindata.inv_transform(h_rec[i].view(1, 3, 128, 128)).convert('RGB')
            # print(img.shape)
            img.save(os.path.join('result', f'{i + idx}_h_fake.jpg'))

        if (idx + 1) % 50 == 0:
            print(f'batch: [{idx}/{len(traindata)}]')


if __name__ == '__main__':
    train()
