import numpy as np
import torch
import torch.nn as nn

from .base import Attack, LabelMixin

from .utils import batch_clamp
from .utils import batch_multiply
from .utils import clamp
from .utils import clamp_by_pnorm
from .utils import is_float_or_torch_tensor
from .utils import normalize_by_pnorm
from .utils import rand_init_delta
from core.attacks._util.ac_gan import netG, netD
import torchvision.transforms as transforms
from core.metrics import accuracy

def perturb_iterative(zvar, yvar, predict, generator, discriminator, nb_iter, eps, eps_iter, loss_fn, lamb1, lamb2, delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0):
    """
    Iteratively maximize the loss over the input. It is a shared method for iterative attacks.
    Arguments:
        xvar (torch.Tensor): input data.
        yvar (torch.Tensor): input labels.
        predict (nn.Module): forward pass function.
        nb_iter (int): number of iterations.
        eps (float): maximum distortion.
        eps_iter (float): attack step size.
        loss_fn (nn.Module): loss function.
        delta_init (torch.Tensor): (optional) tensor contains the random initialization.
        minimize (bool): (optional) whether to minimize or maximize the loss.
        ord (int): (optional) the order of maximum distortion (inf or 2).
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
    Returns:
        torch.Tensor containing the perturbed input,
        torch.Tensor containing the perturbation
    """
    device = zvar.device
    batch_size = zvar.size(0)
    if delta_init is not None:
        delta = delta_init
    else:
        delta = torch.zeros_like(zvar)
    xvar = generator(zvar, yvar)
    delta.requires_grad_()
    for ii in range(nb_iter):
        x = generator(zvar + delta, yvar)
        outputs = predict(transforms.Resize(32)(x))
        loss = loss_fn(outputs, yvar)
        norm_loss = torch.mean(torch.relu(torch.abs(delta) - eps))

        _, plabels = discriminator(x)
        val_loss = torch.nn.functional.nll_loss(plabels, yvar)

        loss = loss - lamb1 * norm_loss - lamb2 * val_loss

        if minimize:
            loss = -loss

        loss.backward()

        grad_sign = delta.grad.data.sign()
        delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
        delta.grad.data.zero_()

        # if (ii+1) % 10 == 0:
        #     x_adv = generator(zvar + delta, yvar)
        #     with torch.no_grad():
        #         out = predict(transforms.Resize(32)(x_adv))
        #     print('iter:{}  acc:{:.5f}%'.format(ii+1, accuracy(yvar, out)*100))

    x_adv = generator(zvar + delta, yvar)
    r_adv = x_adv - xvar
    return x_adv, r_adv


class UsongAttack(Attack, LabelMixin):

    def __init__(self, predict, generator, discriminator, loss_fn=None, eps=0.1, nb_iter=20, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, rand_init_type='uniform'):
        super(UsongAttack, self).__init__(predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.rand_init_type = rand_init_type
        self.ord = np.inf
        self.targeted = targeted
        # self.lamb1 = lamb1
        # self.lamb2 = lamb2
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

        self.netG = generator
        self.netD = discriminator


    def perturb(self, z, y=None, lamb1=100, lamb2=100):
        """
        Given examples (z, y), returns their adversarial counterparts with an attack length of eps.
        Arguments:
            z (torch.Tensor): input tensor.
            y (torch.Tensor): label tensor.
                - if None and self.targeted=False, compute y as predicted
                labels.
                - if self.targeted=True, then y must be the targeted labels.
        Returns:
            torch.Tensor containing perturbed inputs,
            torch.Tensor containing the perturbation
        """
        z, y = self._verify_and_process_inputs(z, y)

        delta = torch.zeros_like(z)
        delta = nn.Parameter(delta)
        if self.rand_init:
            if self.rand_init_type == 'uniform':
                rand_init_delta(
                    delta, z, self.ord, self.eps, self.clip_min, self.clip_max)
                delta.data = clamp(
                    z + delta.data, min=self.clip_min, max=self.clip_max) - z
            elif self.rand_init_type == 'normal':
                delta.data = 0.001 * torch.randn_like(z) # initialize as in TRADES
            else:
                raise NotImplementedError('Only rand_init_type=normal and rand_init_type=uniform have been implemented.')
        
        x_adv, r_adv = perturb_iterative(
            z, y, self.predict, self.netG, self.netD, nb_iter=self.nb_iter, eps=self.eps, eps_iter=self.eps_iter, loss_fn=self.loss_fn,
            minimize=self.targeted, clip_min=self.clip_min, clip_max=self.clip_max, delta_init=delta,
            lamb1=lamb1, lamb2=lamb2
        )

        return x_adv.data, r_adv.data

