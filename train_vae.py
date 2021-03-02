# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/21 22:18
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

    traindata = HeadVaeData(args.vae_data_path, args.trainxml)
    train_loader = DataLoader(traindata, batch_size=args.vae_batch_size, shuffle=True, num_workers=1)

    # print(traindata[0])

    f_vae = VAE(zsize=args.vae_zsize, layer_count=5)
    f_vae.cuda()
    f_vae.train()
    f_vae.weight_init(mean=0, std=0.02)

    f_vae_optimizer = optim.Adam(f_vae.parameters(), lr=args.vae_lr, betas=(0.5, 0.999), weight_decay=1e-5)
    f_vae.train()

    h_vae = VAE(zsize=args.vae_zsize, layer_count=5)
    h_vae.cuda()
    h_vae.train()
    h_vae.weight_init(mean=0, std=0.02)

    h_vae_optimizer = optim.Adam(h_vae.parameters(), lr=args.vae_lr, betas=(0.5, 0.999), weight_decay=1e-5)
    h_vae.train()

    sample1 = torch.randn(128, args.vae_zsize).view(-1, args.vae_zsize, 1, 1)

    print('Start training...')

    for epoch in range(args.vae_epochs):
        if (epoch + 1) % 8 == 0:
            for group in f_vae_optimizer.param_groups:
                group['lr'] *= 0.25

            for group in h_vae_optimizer.param_groups:
                group['lr'] *= 0.25

        f_rec_loss, h_rec_loss = 0, 0
        f_kl_loss, h_kl_loss = 0, 0

        for idx, (f_x, h_x) in enumerate(train_loader):
            f_x, h_x = f_x.cuda(), h_x.cuda()

            f_vae.zero_grad()
            rec, mu, logvar = f_vae(f_x)

            loss_re, loss_kl = loss_function(rec, f_x, mu, logvar)
            (loss_re + loss_kl).backward()
            f_vae_optimizer.step()
            f_rec_loss += loss_re.item()
            f_kl_loss += loss_kl.item()
            f_loss = (loss_re.item() + loss_kl.item()) / len(f_x)

            h_vae.zero_grad()
            rec, mu, logvar = h_vae(h_x)

            loss_re, loss_kl = loss_function(rec, h_x, mu, logvar)
            (loss_re + loss_kl).backward()
            h_vae_optimizer.step()
            h_rec_loss += loss_re.item()
            h_kl_loss += loss_kl.item()
            h_loss = (loss_re.item() + loss_kl.item()) / len(h_x)

            if (idx + 1) % 50 == 0:
                print(f'epoch: [{epoch}/{args.vae_epochs}] batch: [{idx}/{len(traindata) // args.vae_batch_size}] '
                      f'f_loss: {f_loss}, h_loss: {h_loss}')

        f_loss = (f_kl_loss + f_rec_loss) / len(traindata)
        h_loss = (h_kl_loss + h_rec_loss) / len(traindata)

        torch.save(f_vae.state_dict(), os.path.join(args.model_path,
                                                    f'{"f_vae"}_{args.vae_zsize}_{epoch}.pth'))
        torch.save(h_vae.state_dict(), os.path.join(args.model_path,
                                                    f'{"h_vae"}_{args.vae_zsize}_{epoch}.pth'))

        print(f'epoch: {epoch}, f_loss: {f_loss}, h_loss: {h_loss}, '
              f'f_rec_loss: {f_rec_loss / len(traindata)}, f_kl_loss: {f_kl_loss / len(traindata)}, '
              f'h_rec_loss: {h_rec_loss / len(traindata)}, h_kl_loss: {h_kl_loss / len(traindata)}')


if __name__ == '__main__':
    train()
