# Author: BeiYu
# Github: https://github.com/beiyuouo
# Date  : 2021/2/21 22:18
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


def train():
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    traindata = HeadGanData(args.gan_data_path, args.train_txt)
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
    D = Discriminator(img_shape=(3, 128, 128))
    G.cuda()
    D.cuda()
    G.train()
    D.train()

    sample1 = torch.randn(128, args.vae_zsize).view(-1, args.vae_zsize, 1, 1)

    print('Start training...')

    criterion_D = F.binary_cross_entropy_with_logits
    criterion_G = nn.L1Loss()
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.gan_lr, betas=(args.gan_beta, 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.gan_lr, betas=(args.gan_beta, 0.999))

    real_ = 1
    fake_ = 0

    for epoch in range(args.gan_epochs):
        if (epoch + 1) % 5 == 0:
            for group in optimizer_G.param_groups:
                group['lr'] /= 3

            for group in optimizer_D.param_groups:
                group['lr'] /= 3

        total_loss_G = 0.0
        total_loss_D = 0.0
        for i, (a_img, f_img, h_img) in enumerate(train_loader):
            f_img, h_img, a_img = f_img.cuda(), h_img.cuda(), a_img.cuda()
            z_f, z_h = f_vae.get_z(f_img), h_vae.get_z(h_img)
            # print(z_f.shape, z_h.shape)

            minibs = f_img.size(0)
            fake_img = G(z_f, z_h)
            # print(fake_img.shape)
            optimizer_D.zero_grad()

            fake_label = D(fake_img)
            fake_label = torch.mean(fake_label, dim=(1, 2, 3))
            labels = torch.full((minibs,), fake_, device=args.device, dtype=torch.float)
            # print(fake_label.shape, labels.shape)
            loss_D_fake = criterion_D(fake_label, labels)
            # print(loss_D_fake.item())

            # D.zero_grad()
            real_label = D(a_img)
            real_label = torch.mean(real_label, dim=(1, 2, 3))
            labels = torch.full((minibs,), real_, device=args.device, dtype=torch.float)
            loss_D_real = criterion_D(real_label, labels)
            loss_D = loss_D_fake + loss_D_real
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            optimizer_G.zero_grad()
            fake_img = G(z_f, z_h)
            loss_G = criterion_G(fake_img, a_img)
            loss_G.backward()
            optimizer_G.step()

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()

            if (i + 1) % 50 == 0:
                print(f'Epoch: [{epoch}/{args.gan_epochs}] batch: [{i}/{len(traindata) // args.gan_batch_size}] '
                      f'loss_D: {loss_D.item() / minibs}, loss_G: {loss_G.item() / minibs}')

        torch.save(G.state_dict(), os.path.join(args.model_path,
                                                f'{"G"}_{epoch}.pth'))
        torch.save(D.state_dict(), os.path.join(args.model_path,
                                                f'{"D"}_{epoch}.pth'))
        print(f'Epoch {epoch}: -> loss_G: {total_loss_G / len(traindata)}, loss_D: {total_loss_D / len(traindata)}')


if __name__ == '__main__':
    train()
