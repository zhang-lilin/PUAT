import torch
from torch import nn
from torchvision import transforms
from .resnet import resnet, Normalization
from .wideresnet import wideresnet
from .wideresnetwithswish import wideresnetwithswish


def create_model(args, info, device, logger):

    augmentation = True if 'puat' in args.method else False
    name, normalize, use_augmentation  = args.model, args.normalize, args.augment

    if 'resnet' in name and 'preact' not in name:
        backbone = resnet(name, num_classes=info['num_classes'], device=device,
                          normalize=normalize, mean=info['mean'], std=info['std'])
    elif 'wrn' in name and 'swish' not in name:
        backbone = wideresnet(name, logger, num_classes=info['num_classes'], device=device,
                              normalize=normalize, mean=info['mean'], std=info['std'])
    elif 'wrn' in name and 'swish' in name:
        backbone = wideresnetwithswish(name, logger, num_classes=info['num_classes'], device=device,
                                       normalize=normalize, mean=info['mean'], std=info['std'])
    else:
        raise ValueError('Invalid model name {}!'.format(name))

    if augmentation:
        if 'cifar100' in info['data']:
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

        elif 'imagenet32' in info['data']:
            if use_augmentation == 'base':
                train_transform = transforms.Compose(
                    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), ])
            else:
                print("No augmentation used for netC.")

        else:
            raise NotImplementedError

        trans = Trans(train_transform)
        backbone = torch.nn.Sequential(trans, backbone)

    backbone = backbone.to(device)
    return backbone


class Trans(nn.Module):
    def __init__(self, train_transform):
        super(Trans, self).__init__()
        self.trans_train = train_transform

    def forward(self, x):
        if self.training:
            return self.trans_train(x)
        else:
            return x

