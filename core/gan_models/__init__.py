from .discriminator import SNResNetProjectionDiscriminator as discriminator
from .generator import ResNetGenerator as generator
from .generator import Atk_Generator as attacker
import torch
from torch import nn
from torchvision import transforms

from .discriminator import discriminator_dict
from .generator import generator_dict, attacker_dict
import torch.nn.functional as F
import numpy as np

hw_dict = {
    "cifar10": (32, 3, 10),
    "cifar100": (32, 3, 100),
    # "stl10": (96, 3, 10),
    "svhn": (32, 3, 10),
    # "mnist": (28, 1, 10),
    # "fashionmnist": (32, 1, 10),
    # "tiny-imagenet": (64, 3, 200),
    "tiny-imagenet32": (32, 3, 10),

    "imagenet32": (32, 3, 10),
}
actvn_dict = {
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "lrelu": lambda: nn.LeakyReLU(0.2),
}
norm_dict = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    "svhn": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


def get_optimizer(params, opt_name, lr, beta1, beta2, weight_decay):
    if opt_name.lower() == "adam":
        optim = torch.optim.Adam(params, lr, betas=(beta1, beta2), weight_decay=weight_decay)
    elif opt_name.lower() == "nesterov":
        optim = torch.optim.SGD(
            params, lr, momentum=beta1, weight_decay=weight_decay, nesterov=True
        )
    return optim


def get_generator_optimizer(arg, base_dataset):
    module = generator_dict[arg.g_model_name.lower()]
    hw, c, nlabel = hw_dict[base_dataset.lower()]
    actvn = actvn_dict[arg.g_actvn]()
    if arg.g_norm:
        (mean, std) = norm_dict[base_dataset.lower()]
    else:
        (mean, std) = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    G = module(
        # nfilter_max=FLAGS.g_nfilter_max,
        num_classes = nlabel,
        num_input_channels = c,
        num_filter = arg.g_nfilter,
        z_dim = arg.g_z_dim,
        bottom_width = 4,
        activation_fn = actvn,
        mean = mean,
        std = std,
    )
    optim = get_optimizer(
        G.parameters(), arg.g_optim, arg.g_lr, arg.g_beta1, arg.g_beta2, arg.g_weight_decay
    )
    G = G.to(arg.device)
    return G, optim


def get_attacker_optimizer(arg, base_dataset):
    module = attacker_dict[arg.g_model_name.lower()]
    _, _, nlabel = hw_dict[base_dataset.lower()]
    actvn = actvn_dict[arg.a_actvn]()
    A = module(
        z_dim=arg.g_z_dim,
        num_classes = nlabel,
        y_embed_size=arg.a_embed_size,
        epsilon=arg.a_clip,
        activation_fn = actvn,
    )
    optim = get_optimizer(
        A.parameters(), arg.a_optim, arg.a_lr, arg.a_beta1, arg.a_beta2, arg.a_weight_decay
    )
    A = A.to(arg.device)
    return A, optim


def get_discriminator_optimizer(arg, base_dataset, aug=False, device_ids=None):
    module = discriminator_dict[arg.d_model_name.lower()]
    hw, c, nlabel = hw_dict[base_dataset]
    if arg.d_norm:
        (mean, std) = norm_dict[base_dataset.lower()]
    else:
        (mean, std) = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    D = module(
        num_classes = nlabel,
        num_input_channels = c,
        nfilter = arg.d_nfilter,
        activation_fn = actvn_dict[arg.d_actvn](),
        mean = mean,
        std = std,
    )

    if aug:
        if base_dataset == 'cifar100':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
            ])
        elif base_dataset == 'cifar10':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(0.5),
            ])
        elif base_dataset == 'svhn':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=2),
            ])
        elif base_dataset == 'tiny-imagenet32':
            train_transform = transforms.RandomCrop(32, padding=2)
        else:
            print(base_dataset)
            raise NotImplementedError

        D = Trans_D(train_transform, D)

    optim = get_optimizer(
        D.parameters(), arg.d_optim, arg.d_lr, arg.d_beta1, arg.d_beta2, arg.d_weight_decay
    )

    # if device_ids is not None:
    #     D = torch.nn.DataParallel(D, device_ids=device_ids)
    #     print('D: {}'.format(device_ids))
    # else:
    #     D = torch.nn.DataParallel(D)
    D = D.to(arg.device)

    return D, optim


class Trans_D(nn.Module):
    def __init__(self, train_transform, discriminator):
        super(Trans_D, self).__init__()
        self.trans_train = train_transform
        self.dis = discriminator

    def forward(self, x, y=None):
        if self.training:
            x = self.trans_train(x)
            return self.dis(x, y)
        else:
            return self.dis(x, y)