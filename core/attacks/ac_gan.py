import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.distributions import laplace
from torch.distributions import uniform
from torch.nn.modules.loss import _Loss
from tqdm import tqdm

from core.utils import seed


class Generator(nn.Module):

    def __init__(self, num_classes = 10):
        super(Generator, self).__init__()

        # input 100*1*1
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
                                    nn.ReLU(True))

        # input 512*4*4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(True))
        # input 256*8*8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(True))
        # input 128*16*16
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(True))
        # input 64*32*32
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                                    nn.Tanh())
        # output 3*64*64

        self.embedding = nn.Embedding(num_classes, 100)

    def forward(self, noise, label):
        label_embedding = self.embedding(label)
        x = torch.mul(noise, label_embedding)
        x = x.view(-1, 100, 1, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = x * 0.5 + 0.5
        return x


class Discriminator(nn.Module):

    def __init__(self, num_classes = 10):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

        # input 3*64*64
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))

        # input 64*32*32
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 128*16*16
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2, True),
                                    nn.Dropout2d(0.5))
        # input 256*8*8
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2, True))
        # input 512*4*4
        self.validity_layer = nn.Sequential(nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                                            nn.Sigmoid())

        self.label_layer = nn.Sequential(nn.Conv2d(512, num_classes + 1, 4, 1, 0, bias=False),
                                         nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = (x - 0.5) / 0.5
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        validity = self.validity_layer(x)
        plabel = self.label_layer(x)

        validity = validity.view(-1)
        plabel = plabel.view(-1, self.num_classes + 1)

        return validity, plabel



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



class AC_GAN_Trainer():
    def __init__(self, info, args):
        super(AC_GAN_Trainer, self).__init__()

        seed(args.seed)
        self.device = device = args.device
        self.num_classes = info['num_classes']
        self.netD = Discriminator(info['num_classes']).to(device)
        self.optD = optim.Adam(self.netD.parameters(), 0.0002, betas=(0.5, 0.999))
        self.netG = Generator(info['num_classes']).to(device)
        self.optG = optim.Adam(self.netG.parameters(), 0.0002, betas=(0.5, 0.999))

        self.params = args

        self.real_labels = 0.7 + 0.5 * torch.rand(10, device=device)
        self.fake_labels = 0.3 * torch.rand(10, device=device)


    def train(self, dataloader, epoch=0, verbose=False, logger=None):
        """
        Run one epoch of training.
        """
        validity_loss = nn.BCELoss()
        self.netG.train(), self.netD.train()
        device = self.device
        for idx, (images, labels) in tqdm(enumerate(dataloader, 0), desc='Epoch {}: '.format(epoch), disable=not verbose):
            images, labels = images.to(device), labels.to(device)
            global_step = (epoch - 1) * len(dataloader) + idx

            batch_size = images.size(0)
            labels = labels.to(device)
            images = images.to(device)

            real_label = self.real_labels[idx % 10]
            fake_label = self.fake_labels[idx % 10]

            fake_class_labels = self.num_classes * torch.ones((batch_size,), dtype=torch.long, device=device)
            if idx % 25 == 0:
                real_label, fake_label = fake_label, real_label

            # ---------------------
            #         disc
            # ---------------------

            self.optD.zero_grad()

            # real
            validity_label = torch.full((batch_size,), real_label, device=device)
            pvalidity, plabels = self.netD(images)

            errD_real_val = validity_loss(pvalidity, validity_label)
            errD_real_label = F.nll_loss(plabels, labels)

            errD_real = errD_real_val + errD_real_label
            errD_real.backward()

            D_x = pvalidity.mean().item()

            # fake
            noise = torch.randn(batch_size, 100, device=device)
            sample_labels = torch.randint(0, self.num_classes, (batch_size,), device=device, dtype=torch.long)

            fakes = self.netG(noise, sample_labels)

            validity_label.fill_(fake_label)
            pvalidity, plabels = self.netD(fakes.detach())

            errD_fake_val = validity_loss(pvalidity, validity_label)
            errD_fake_label = F.nll_loss(plabels, fake_class_labels)

            errD_fake = errD_fake_val + errD_fake_label
            errD_fake.backward()

            D_G_z1 = pvalidity.mean().item()

            # finally update the params!
            errD = errD_real + errD_fake

            self.optD.step()

            # ------------------------
            #      gen
            # ------------------------

            self.optG.zero_grad()

            noise = torch.randn(batch_size, 100, device=device)
            sample_labels = torch.randint(0, self.num_classes, (batch_size,), device=device, dtype=torch.long)

            validity_label.fill_(1)

            fakes = self.netG(noise, sample_labels)
            pvalidity, plabels = self.netD(fakes)

            errG_val = validity_loss(pvalidity, validity_label)
            errG_label = F.nll_loss(plabels, sample_labels)

            errG = errG_val + errG_label
            errG.backward()

            D_G_z2 = pvalidity.mean().item()

            self.optG.step()

            # print(
            #     " D_x: [{:.4f}] D_G: [{:.4f}/{:.4f}] G_loss: [{:.4f}] D_loss: [{:.4f}] D_label: [{:.4f}] "
            #     .format(D_x, D_G_z1, D_G_z2, errG, errD,
            #             errD_real_label + errD_fake_label + errG_label))

            if logger is not None:
                logger.add("training_d", "loss", errD.item(), global_step)
                logger.add("training_d", "dreal", D_x, global_step)
                logger.add("training_d", "dfake", D_G_z1, global_step)
                logger.add("training_d", "dlabel", (errD_real_label + errD_fake_label + errG_label).item(), global_step)
                logger.add("training_g", "loss", errG.item(), global_step)
                logger.add("training_g", "dfake", D_G_z2, global_step)

        if logger is not None:
            logger.log_info(global_step, ["training_d", "training_g"])


    def eval_generator(self, num_per_class=10):
        device = self.device
        noise_ = torch.randn(num_per_class, 100).to(device)
        noise = torch.cat([noise_ for _ in range(self.num_classes)], 0).to(device)
        self.netG.eval()
        with torch.no_grad():
            s = torch.Size([num_per_class])
            test_label = torch.cat([torch.full(s, k) for k in range(self.num_classes)], 0).to(device)
            x_fake = self.netG(noise, test_label)
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




