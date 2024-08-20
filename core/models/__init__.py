import numpy as np
import torch
from torch import nn
from core.data.cifar10s import IDBH, CIFAR10Policy, CutoutDefault, MultiDataTransform, RandAugment
# from .resnet import Normalization
from torchvision import transforms

from .preact_resnet import preact_resnet
from .resnet import resnet, Normalization
from .wideresnet import wideresnet

from .preact_resnetwithswish import preact_resnetwithswish
from .wideresnetwithswish import wideresnetwithswish
from .ti_wideresnetwithswish import ti_wideresnetwithswish

from core.data import DATASETS


MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 
          'preact-resnet18', 'preact-resnet34', 'preact-resnet50', 'preact-resnet101', 
          'wrn-28-10', 'wrn-32-10', 'wrn-34-10', 'wrn-34-20', 
          'preact-resnet18-swish', 'preact-resnet34-swish',
          'wrn-28-10-swish', 'wrn-34-20-swish', 'wrn-70-16-swish']


def create_model(args, info, device):

    name, normalize, use_augmentation, use_consistency = args.model, args.normalize, args.augment, args.consistency

    dataset = info['data']
    num_classes = info['num_classes']
    mean, std = info['mean'], info['std']
    print(f'Creat Model for dataset {dataset}({num_classes})')

    if 'wrn' in name and 'swish' in name:
        backbone = wideresnetwithswish(name, dataset=dataset, num_classes=num_classes, device=device,
                                       mean=mean, std=std)
    else:
        raise ValueError('Invalid model name {}!'.format(name))


    if args.puat:
        if 'svhn' in info['data']:
            train_transform = None

        elif 'cifar100' in info['data']:
            if use_augmentation == 'base':
                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(15),
                ])
            else:
                print("No augmentation used for netC.")
        elif 'cifar10' in info['data']:
            if use_augmentation == 'base':
                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(0.5),
                ])
            else:
                print("No augmentation used for netC.")

        elif 'tiny-imagenet' in info['data']:
            if use_augmentation == 'base':
                train_transform = transforms.Compose(
                    [transforms.Resize(32),
                     transforms.RandomCrop(32, padding=2),
                     transforms.RandomHorizontalFlip(), ])
            else:
                print("No augmentation used for netC.")

        elif 'imagenet32' in info['data']:
            if use_augmentation == 'base':
                train_transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), ])
            else:
                print("No augmentation used for netC.")

        else:
            raise NotImplementedError

        trans = Trans(train_transform)

        if normalize:
            model = torch.nn.Sequential(Normalization(info['mean'], info['std']), trans, backbone)
        else:
            model = torch.nn.Sequential(trans, backbone)

    else:
        if normalize:
            model = torch.nn.Sequential(Normalization(info['mean'], info['std']), backbone)
        else:
            model = torch.nn.Sequential(backbone)

    model = model.to(device)
    return model


class Trans(nn.Module):
    def __init__(self, train_transform):
        super(Trans, self).__init__()
        self.trans_train = train_transform

    def forward(self, x):
        if self.training and self.trans_train:
            return self.trans_train(x)
        else:
            return x
