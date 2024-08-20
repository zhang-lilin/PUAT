import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
import os
from core.attacks import create_attack
from core.metrics import accuracy
from core.models import create_model
from core.gan_models.wapper import get_generator_optimizer, get_discriminator_optimizer
from .ac_gan import netD, netG
from core.utils import seed

hw_dict = {
    "cifar10": (32, 3, 10),
    "cifar100": (32, 3, 100),
    "stl10": (96, 3, 10),
    "svhn": (32, 3, 10),
    "mnist": (28, 1, 10),
    "fashionmnist": (32, 1, 10),
    "tinyimagenet": (64, 3, 10),
    "tinyimagenet32": (32, 3, 10),
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer(object):

    def __init__(self, args):
        super(Trainer, self).__init__()
        self.hw, self.nc, self.num_classes = hw_dict[args.data.lower()]
        self.params = args

        seed(args.seed)
        # self.netG, self.optG = get_generator_optimizer(self.params, self.params.data)
        # self.netD, self.optD = get_discriminator_optimizer(self.params, self.params.data)
        self.netG = netG(args.g_z_dim + self.num_classes, args.g_nfilter, self.nc).to(device)
        self.netD = netD(args.d_nfilter, self.nc, self.num_classes).to(device)
        self.optD = torch.optim.Adam(self.netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        self.optG = torch.optim.Adam(self.netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))


    def train(self, dataloader, epoch=0, logger=None, verbose=False):
        """
        Run one epoch of training.
        """
        self.netG.train()
        self.netD.train()

        c_criterion = nn.CrossEntropyLoss(reduction='mean')
        d_iter = 5
        
        batch_size = self.params.batch_size
        data = get_itr(dataloader)
        # update_iter = 0

        # for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
        for update_iter in tqdm(range(1, len(dataloader)+1), desc='Epoch {}: '.format(epoch), disable=not verbose):
            # update_iter += 1
            global_step = (epoch - 1) * len(dataloader) + update_iter

            """Discriminator"""
            for _ in range(d_iter):
                img, label = data.__next__()
                img, label = img.to(device), label.to(device)
                self.optD.zero_grad()
                s_output, c_output = self.netD(img)
                s_errD_real = torch.mean(torch.relu(1.0 - s_output))
                c_errD_real = c_criterion(c_output, label)

                errD_real = s_errD_real + c_errD_real
                errD_real.backward()

                # label_rand = torch.randint(self.num_classes, torch.Size([batch_size])).to(device)
                # label_rand_onehot = self.one_hot_(label_rand)
                # z_rand = torch.randn(batch_size, self.params.g_z_dim).to(device)
                # noise = torch.cat([label_rand_onehot, z_rand], dim=1)
                # noise = noise.resize_(batch_size, self.params.g_z_dim + self.num_classes, 1, 1)

                sample_z = torch.randn(self.params.batch_size, self.params.g_z_dim).to(device)
                fake = self.netG(sample_z, label)
                s_output, c_output = self.netD(fake.detach())
                s_errD_fake = torch.mean(torch.relu(1.0 + s_output))
                c_errD_fake = c_criterion(c_output, label)

                errD_fake = s_errD_fake + c_errD_fake
                errD_fake.backward()
                self.optD.step()
            loss_D = errD_fake + errD_real

            """Generator"""
            img, label = data.__next__()
            img, label = img.to(device), label.to(device)
            self.optG.zero_grad()
            s_output, c_output = self.netD(fake)
            s_errG = - torch.mean(s_output)
            c_errG = c_criterion(c_output, label)

            errG = s_errG + c_errG
            errG.backward()
            self.optG.step()
            loss_G = errG

            if logger is not None:
                logger.add("training_d", "loss", loss_D.item(), global_step)
                logger.add("training_d", "c_real", (c_errD_real).item(), global_step)
                logger.add("training_d", "c_fake", (c_errD_fake).item(), global_step)
                logger.add("training_d", "s_real", (s_errD_real).item(), global_step)
                logger.add("training_d", "s_fake", (s_errD_fake).item(), global_step)

                logger.add("training_g", "loss", loss_G.item(), global_step)
                logger.add("training_g", "s_fake", s_errG.item(), global_step)
                logger.add("training_g", "c_fake", c_errG.item(), global_step)
        if logger is not None:
            logger.log_info(global_step, ["training_d", "training_g"])

    def one_hot_(self, label):
        return torch.eye(self.num_classes).to(device).index_select(dim=0, index=label)
    
    # def eval_generator(self, z_rand_=None, num_per_class=10):
    #     l_ = torch.eye(self.num_classes)
    #     if z_rand_ is None:
    #         z_rand_ = torch.randn(num_per_class, self.params.g_z_dim).to(device)
    #     z_rand = torch.cat([z_rand_ for _ in range(self.num_classes)], 0).to(device)
    #     with torch.no_grad():
    #         label = torch.cat([torch.cat([l_[k].unsqueeze(0) for _ in range(self.num_classes)], 0) for k in range(self.num_classes)], 0).to(device)
    #         noise = torch.cat([label, z_rand], dim=1).resize_(num_per_class * self.num_classes,
    #                               self.params.g_z_dim + self.num_classes, 1, 1)
    #         fake = self.netG(noise)
    #     self.netG.train()
    #     return fake

    def eval_generator(self, z_rand_=None, num_per_class=10):
        if z_rand_ is None:
            test_z_ = torch.randn(num_per_class, self.params.g_z_dim).to(device)
        else:
            test_z_ = z_rand_
        test_z = torch.cat([test_z_ for _ in range(self.num_classes)], 0).to(device)
        self.netG.eval()
        with torch.no_grad():
            s = torch.Size([num_per_class])
            test_label = torch.cat([torch.full(s, k) for k in range(self.num_classes)], 0).to(device)
            x_fake = self.netG(test_z, test_label)
        self.netG.train()
        return x_fake

    def save_model(self, path, epoch):
        """
        Save model weights and optimizer.
        """
        torch.save({
            'netG': self.netG.state_dict(),
            'optimizer_G': self.optG.state_dict(),
            'netD': self.netD.state_dict(),
            'optimizer_D': self.optD.state_dict(),
            'epoch': epoch
        }, path)

    def load_model(self, path):
        """
        load model weights and optimizer.
        """
        checkpoint = torch.load(path)
        self.netG.load_state_dict(checkpoint['netG'])
        self.optG.load_state_dict(checkpoint['optimizer_G'])
        self.netD.load_state_dict(checkpoint['netD'])
        self.optD.load_state_dict(checkpoint['optimizer_D'])

        return checkpoint['epoch']


def get_itr(loader, infinity=True):
    if infinity is True:
        while True:
            for img, labels in loader:
                yield img, labels
    else:
        for img, labels in loader:
            yield img, labels