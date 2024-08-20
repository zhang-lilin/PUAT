import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.nn import init
from typing import Tuple, Union


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
SVHN_MEAN = (0.5, 0.5, 0.5)
SVHN_STD = (0.5, 0.5, 0.5)


class ConditionalBatchNorm2d(nn.BatchNorm2d):

    """Conditional Batch Normalization"""

    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=False,
        track_running_stats=True,
    ):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight=None, bias=None, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):
    def __init__(
        self,
        num_classes,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=True,
    ):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c=None, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(
            input, weight, bias
        )


def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode="bilinear")


class Block(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        h_ch=None,
        ksize=3,
        pad=1,
        activation=F.relu,
        upsample=False,
        num_classes=0,
    ):
        super(Block, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = nn.Conv2d(h_ch, out_ch, ksize, 1, pad)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(num_classes=num_classes, num_features=in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(num_classes=num_classes, num_features=h_ch)
        else:
            self.b1 = nn.BatchNorm2d(num_features=in_ch)
            self.b2 = nn.BatchNorm2d(num_features=h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, gain=math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
                h = self.c_sc(h)
            else:
                h = self.c_sc(x)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if y is not None:
            h = self.b1(input=x, c=y, **kwargs)
        else:
            h = self.b1(input=x)
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))


class ResNetGenerator(nn.Module):
    def __init__(self,
                 num_classes: int = 10,
                 num_input_channels: int = 3,
                 num_filter: int = 64,
                 z_dim: int = 256,
                 bottom_width: int = 4,
                 activation_fn: nn.Module = nn.ReLU,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 ):
        super(ResNetGenerator, self).__init__()
        self.num_features = num_features = num_filter
        self.dim_z = z_dim
        self.bottom_width = bottom_width
        self.activation = activation = activation_fn
        self.num_classes = num_classes

        width_coe = 8
        self.l1 = nn.Linear(self.dim_z, width_coe * num_features * bottom_width ** 2)

        self.block2 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block3 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block4 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.b7 = nn.BatchNorm2d(num_features * width_coe)
        self.conv7 = nn.Conv2d(num_features * width_coe, num_input_channels, 1, 1)

        self._initialize()
        self.normalization_used = {}
        self.normalization_used['mean'] = torch.tensor(list(mean)).reshape(1, 3, 1, 1)
        self.normalization_used['std'] = torch.tensor(list(std)).reshape(1, 3, 1, 1)


    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.conv7.weight.data)


    def _inverse_normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return inputs * std + mean


    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in [2, 3, 4]:
            h = getattr(self, "block{}".format(i))(h, y, **kwargs)
        h = self.activation(self.b7(h))
        h = torch.tanh(self.conv7(h))
        x_g = self._inverse_normalize(h)
        return x_g


class ResNetGenerator64(nn.Module):

    def __init__(self,
                 num_classes: int = 200,
                 num_input_channels: int = 3,
                 num_filter: int = 64,
                 z_dim: int = 256,
                 bottom_width: int = 4,
                 activation_fn: nn.Module = nn.ReLU,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,

                 ):
        super(ResNetGenerator64, self).__init__()
        self.num_features = num_features = num_filter
        self.dim_z = z_dim
        self.bottom_width = bottom_width
        self.activation = activation = activation_fn
        self.num_classes = num_classes = num_classes

        width_coe = 8
        self.l1 = nn.Linear(self.dim_z, width_coe * num_features * bottom_width ** 2)
        self.block1 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block2 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block3 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block4 = Block(
            num_features * width_coe,
            num_features * width_coe,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.b7 = nn.BatchNorm2d(num_features * width_coe)
        self.conv7 = nn.Conv2d(num_features * width_coe, 3, 1, 1)
        self._initialize()
        self.normalization_used = {}
        self.normalization_used['mean'] = torch.tensor(list(mean)).reshape(1, 3, 1, 1)
        self.normalization_used['std'] = torch.tensor(list(std)).reshape(1, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.conv7.weight.data)

    def _inverse_normalize(self, inputs):
        mean = self.normalization_used['mean'].to(inputs.device)
        std = self.normalization_used['std'].to(inputs.device)
        return inputs * std + mean

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in [1, 2, 3, 4]:
            h = getattr(self, "block{}".format(i))(h, y, **kwargs)
        h = self.activation(self.b7(h))
        h = torch.tanh(self.conv7(h))
        x_g = self._inverse_normalize(h)
        return x_g


class ResNetGenerator96(nn.Module):
    def __init__(
        self,
        z_dim=256,
        n_label=10,
        im_size=32,
        im_chan=3,
        embed_size=256,
        nfilter=64,
        nfilter_max=512,
        actvn=F.relu,
        distribution="normal",
        bottom_width=6,
    ):
        super(ResNetGenerator96, self).__init__()
        self.num_features = num_features = nfilter
        self.dim_z = z_dim
        self.bottom_width = bottom_width
        self.activation = activation = actvn
        self.num_classes = num_classes = n_label
        self.distribution = distribution

        width_coe = 8
        self.l1 = nn.Linear(
            self.dim_z, 1 * width_coe * num_features * bottom_width ** 2
        )

        self.block2 = Block(
            num_features * width_coe * 1,
            num_features * width_coe * 1,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block3 = Block(
            num_features * width_coe * 1,
            num_features * width_coe * 1,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block4 = Block(
            num_features * width_coe * 1,
            num_features * width_coe * 1,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.block5 = Block(
            num_features * width_coe * 1,
            num_features * width_coe,
            activation=activation,
            upsample=True,
            num_classes=num_classes,
        )
        self.b7 = nn.BatchNorm2d(num_features * width_coe)
        self.conv7 = nn.Conv2d(num_features * width_coe, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv7.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in [2, 3, 4, 5]:
            h = getattr(self, "block{}".format(i))(h, y, **kwargs)
        h = self.activation(self.b7(h))
        return torch.tanh(self.conv7(h))



class Atk_Generator_(nn.Module):
    def __init__(self,
                 z_dim: int = 256,
                 y_embed_size: int = 256,
                 num_classes: int = 10,
                 epsilon: float = 0.1,
                 activation_fn: nn.Module = nn.ReLU,
                 ):
        super().__init__()
        self.actvn = activation_fn
        self.epsilon = epsilon
        self.z_dim = z_dim
        self.num_classes = num_classes

        self.embedding = nn.Embedding(num_classes, y_embed_size)
        self.l1 = nn.Linear(z_dim + y_embed_size, z_dim + y_embed_size)
        self.bn = nn.BatchNorm1d(z_dim + y_embed_size)
        self.l2 = nn.Linear(z_dim + y_embed_size, z_dim)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.embedding.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.l1.weight.data, gain=math.sqrt(2))
        init.xavier_uniform_(self.l2.weight.data, gain=math.sqrt(2))

    def forward(self, z, y):
        assert z.size(0) == y.size(0)

        yembed = self.embedding(y)
        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)

        yz = torch.cat([z, yembed], dim=1)
        out = self.actvn(self.bn(self.l1(yz)))
        out = self.l2(out)
        out = torch.tanh(out)
        out = out * 0.5 + 0.5
        out = z + self.epsilon * out

        # out = torch.clamp(out, z - self.epsilon, z + self.epsilon)
        # print('A pert_max:{:.5f} max_value:{:.5f} min_value:{:.5f} max_z:{:.5f} min_z:{:.5f}'.format(torch.max(torch.abs(out - z)), torch.max(out), torch.min(out), torch.max(z), torch.min(z)))
        return out



class Atk_Generator__(nn.Module):
    def __init__(self,
                 z_dim: int = 256,
                 y_embed_size: int = 256,
                 bottom_width = 16,
                 num_classes: int = 10,
                 epsilon: float = 0.1,
                 activation_fn: nn.Module = nn.ReLU,
                 ):
        super().__init__()
        self.actvn = activation_fn
        self.epsilon = epsilon
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.bottom_width = bottom_width
        in_ch, out_ch = 1, 1
        ksize, pad = 3, 1

        self.l1 = nn.Linear(self.z_dim, bottom_width ** 2)
        self.c1 = nn.Conv2d(in_ch, out_ch, ksize, 1, pad)
        self.b1 = CategoricalConditionalBatchNorm2d(num_classes=num_classes, num_features=in_ch)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.c1.weight.data)

    def forward(self, z, y):
        assert z.size(0) == y.size(0)

        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.b1(input=h, c=y)
        h = self.c1(self.actvn(h))
        h = torch.tanh(h.view(z.size(0), -1))
        # h = h * 0.5 + 0.5
        z_a = z + self.epsilon * h

        return z_a


class Atk_Generator(nn.Module):
    def __init__(self,
                 z_dim: int = 256,
                 y_embed_size: int = 256,
                 bottom_width = 4,
                 num_classes: int = 10,
                 epsilon: float = 0.1,
                 activation_fn: nn.Module = nn.ReLU,
                 ):
        super().__init__()
        self.actvn = activation_fn
        self.epsilon = epsilon
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.bottom_width = bottom_width
        self.fea = 16
        in_ch, out_ch = 16, 16
        h_ch = 32
        ksize, pad = 3, 1

        self.l1 = nn.Linear(self.z_dim, self.fea * bottom_width ** 2)
        self.block2 = Block(
            in_ch,
            h_ch,
            activation=activation_fn,
            upsample=False,
            num_classes=num_classes,
        )
        self.block3 = Block(
            h_ch,
            out_ch,
            activation=activation_fn,
            upsample=False,
            num_classes=num_classes,
        )
        self.l4 = nn.Linear(self.fea * bottom_width ** 2, self.z_dim)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.l4.weight.data)

    def forward(self, z, y):
        assert z.size(0) == y.size(0)

        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y=y)
        h = self.block3(h, y=y)
        h = self.l4(h.view(z.size(0), -1))
        h = torch.tanh(h)
        z_a = z + self.epsilon * h

        return z_a


class Atk_Generator_deep(nn.Module):
    def __init__(self,
                 z_dim: int = 256,
                 y_embed_size: int = 256,
                 bottom_width = 4,
                 num_classes: int = 10,
                 epsilon: float = 0.1,
                 activation_fn: nn.Module = nn.ReLU,
                 ):
        super().__init__()
        self.actvn = activation_fn
        self.epsilon = epsilon
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.bottom_width = bottom_width
        self.fea = in_ch = out_ch = 32
        h_ch = 2 * in_ch
        ksize, pad = 3, 1

        self.l1 = nn.Linear(self.z_dim, self.fea * bottom_width ** 2)
        self.block2 = Block(
            in_ch,
            h_ch,
            activation=activation_fn,
            upsample=False,
            num_classes=num_classes,
        )
        self.block3 = Block(
            h_ch,
            h_ch,
            activation=activation_fn,
            upsample=False,
            num_classes=num_classes,
        )
        self.block4 = Block(
            h_ch,
            out_ch,
            activation=activation_fn,
            upsample=False,
            num_classes=num_classes,
        )
        self.l4 = nn.Linear(self.fea * bottom_width ** 2, self.z_dim)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.l4.weight.data)

    def forward(self, z, y):
        assert z.size(0) == y.size(0)

        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y=y)
        h = self.block3(h, y=y)
        h = self.block4(h, y=y)
        h = self.l4(h.view(z.size(0), -1))
        # z_a = z + h
        # z_a = h
        h = torch.tanh(h)
        z_a = z + self.epsilon * h

        return z_a


class atk_Generator(nn.Module):
    def __init__(self,
                 z_dim: int = 256,
                 y_embed_size: int = 256,
                 bottom_width=4,
                 num_classes: int = 10,
                 epsilon: float = 0.1,
                 activation_fn: nn.Module = nn.ReLU,
                 ):
        super().__init__()
        self.actvn = activation_fn
        self.epsilon = epsilon
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.bottom_width = bottom_width
        self.fea = 16
        in_ch, out_ch = 16, 16
        h_ch = 32
        ksize, pad = 3, 1

        self.embedding = nn.Embedding(num_classes, y_embed_size)
        self.l1 = nn.Linear(self.z_dim, self.fea * bottom_width ** 2)
        self.block2 = Block(
            in_ch,
            h_ch,
            activation=activation_fn,
            upsample=False,
            num_classes=num_classes,
        )
        self.block3 = Block(
            h_ch,
            out_ch,
            activation=activation_fn,
            upsample=False,
            num_classes=num_classes,
        )
        self.l4 = nn.Linear(self.fea * bottom_width ** 2, self.z_dim)

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.embedding.weight.data)
        init.xavier_uniform_(self.l1.weight.data)
        init.xavier_uniform_(self.l4.weight.data)

    def forward(self, z, y):
        assert z.size(0) == y.size(0)

        yembed = self.embedding(y)
        yembed = yembed / torch.norm(yembed, p=2, dim=1, keepdim=True)
        yz = torch.mul(z, yembed)

        h = self.l1(yz).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y=y)
        h = self.block3(h, y=y)
        h = self.l4(h.view(z.size(0), -1))
        z_a = h
        # h = torch.tanh(h)
        # z_a = z + self.epsilon * h

        return z_a


generator_dict = {
    "resnet_sngan": ResNetGenerator,
    "resnet_sngan64": ResNetGenerator64,
}

attacker_dict = {
    "resnet_sngan": Atk_Generator,
    "resnet_sngan64": Atk_Generator,
}