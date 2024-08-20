import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils
from typing import Tuple, Union

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
SVHN_MEAN = (0.5, 0.5, 0.5)
SVHN_STD = (0.5, 0.5, 0.5)


class Block(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        h_ch=None,
        ksize=3,
        pad=1,
        activation=F.relu,
        downsample=False,
    ):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)


class SNResNetProjectionDiscriminator(nn.Module):
    def __init__(self,
                 num_classes: int = 10,
                 num_input_channels: int = 3,
                 nfilter: int = 64,
                 activation_fn: nn.Module = nn.ReLU,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 ):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features = nfilter
        self.num_classes = num_classes
        self.activation = activation = activation_fn

        width_coe = 8
        self.block1 = OptimizedBlock(num_input_channels, num_features * width_coe)
        self.block2 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block3 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block4 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.l7 = utils.spectral_norm(nn.Linear(num_features * width_coe, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * width_coe)
            )

        self._initialize()
        self.normalization_used = {}
        self.normalization_used['mean'] = torch.tensor(list((0.5, 0.5, 0.5))).reshape(1, 3, 1, 1)
        self.normalization_used['std'] = torch.tensor(list((0.5, 0.5, 0.5))).reshape(1, 3, 1, 1)


    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, "l_y", None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)


    def _normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return (inputs - mean) / std


    def forward(self, x, y=None):
        bs = x.shape[0]
        h = self._normalize(x)
        for i in range(1, 5):
            h = getattr(self, "block{}".format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        else:
            output_list = []
            for i in range(self.num_classes):
                ty = torch.ones([bs,], dtype=torch.long) * i
                toutput = output + torch.sum(
                    self.l_y(ty.to(x.device)) * h, dim=1, keepdim=True
                )
                output_list.append(toutput)
            output = torch.cat(output_list, dim=1)
        return output


class _SNResNetProjectionDiscriminator(nn.Module):
    def __init__(self,
                 num_classes: int = 10,
                 num_input_channels: int = 3,
                 nfilter: int = 64,
                 activation_fn: nn.Module = nn.ReLU,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 ):
        super(_SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features = nfilter
        self.num_classes = num_classes
        self.activation = activation = activation_fn

        width_coe = 8
        self.block1 = OptimizedBlock(num_input_channels, num_features * width_coe)
        self.block2 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block3 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.block4 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            downsample=True,
        )
        self.l7 = utils.spectral_norm(nn.Linear(num_features * width_coe, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * width_coe)
            )

        self._initialize()
        self.normalization_used = {}
        self.normalization_used['mean'] = torch.tensor(list((0.5, 0.5, 0.5))).reshape(1, 3, 1, 1)
        self.normalization_used['std'] = torch.tensor(list((0.5, 0.5, 0.5))).reshape(1, 3, 1, 1)


    def _initialize(self):
        init.xavier_uniform_(self.l7.weight.data)
        optional_l_y = getattr(self, "l_y", None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)


    def _normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return (inputs - mean) / std


    def forward(self, x, y):
        bs = x.shape[0]
        h = self._normalize(x)
        for i in range(1, 5):
            h = getattr(self, "block{}".format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l7(h)
        output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output


discriminator_dict = {
    "resnet_sngan": SNResNetProjectionDiscriminator,
}