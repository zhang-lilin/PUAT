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

def perturb_iterative(zvar, yvar, predict, generator, nb_iter, eps, eps_iter, loss_fn, delta_init=None, minimize=False, ord=np.inf,
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
        if minimize:
            loss = -loss

        loss.backward()
        if ord == np.inf:
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
            delta.data = batch_clamp(eps, delta.data)
            # delta.data = clamp(zvar.data + delta.data, clip_min, clip_max) - zvar.data
        elif ord == 2:
            grad = delta.grad.data
            grad = normalize_by_pnorm(grad)
            delta.data = delta.data + batch_multiply(eps_iter, grad)
            # delta.data = clamp(zvar.data + delta.data, clip_min, clip_max) - zvar.data
            if eps is not None:
                delta.data = clamp_by_pnorm(delta.data, ord, eps)
        else:
            error = "Only ord=inf and ord=2 have been implemented"
            raise NotImplementedError(error)
        delta.grad.data.zero_()

        # if (ii+1) % 10 == 0:
        #     x_adv = generator(zvar + delta, yvar)
        #     with torch.no_grad():
        #         out = predict(transforms.Resize(32)(x_adv))
        #     print('iter:{}  acc:{:.5f}%'.format(ii+1, accuracy(yvar, out)*100))

    x_adv = generator(zvar + delta, yvar)
    r_adv = x_adv - xvar
    return x_adv, r_adv


class GPGDAttack(Attack, LabelMixin):

    def __init__(self, predict, generator, loss_fn=None, eps=0.1, nb_iter=20, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            ord=np.inf, targeted=False, rand_init_type='uniform'):
        super(GPGDAttack, self).__init__(predict, loss_fn, clip_min, clip_max)
        self.eps = eps
        self.nb_iter = nb_iter
        self.eps_iter = eps_iter
        self.rand_init = rand_init
        self.rand_init_type = rand_init_type
        self.ord = ord
        self.targeted = targeted
        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")
        assert is_float_or_torch_tensor(self.eps_iter)
        assert is_float_or_torch_tensor(self.eps)

        self.netG = generator

    def perturb(self, z, y=None):
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
            z, y, self.predict, self.netG, nb_iter=self.nb_iter, eps=self.eps, eps_iter=self.eps_iter, loss_fn=self.loss_fn,
            minimize=self.targeted, ord=self.ord, clip_min=self.clip_min, clip_max=self.clip_max, delta_init=delta
        )

        return x_adv.data, r_adv.data



class LinfGPGDAttack(GPGDAttack):
    """
    PGD Attack with order=Linf
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
        eps_iter (float): attack step size.
        rand_init (bool): (optional) random initialization.    
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        targeted (bool): if the attack is targeted.
        rand_init_type (str): (optional) random initialization type.
    """

    def __init__(
            self, predict, generator, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, rand_init_type='uniform'):
        ord = np.inf
        super(LinfGPGDAttack, self).__init__(
            predict=predict, generator=generator, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, rand_init=rand_init,
            clip_min=clip_min, clip_max=clip_max, targeted=targeted, ord=ord, rand_init_type=rand_init_type)



class L2GPGDAttack(GPGDAttack):
    """
    PGD Attack with order=L2
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (nn.Module): loss function.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
        eps_iter (float): attack step size.
        rand_init (bool): (optional) random initialization.
        clip_min (float): mininum value per input dimension.
        clip_max (float): maximum value per input dimension.
        targeted (bool): if the attack is targeted.
        rand_init_type (str): (optional) random initialization type.
    """

    def __init__(
            self, predict, generator, loss_fn=None, eps=0.3, nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.,
            targeted=False, rand_init_type='uniform'):
        ord = 2
        super(L2GPGDAttack, self).__init__(
            predict=predict, generator=generator, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter, eps_iter=eps_iter, rand_init=rand_init,
            clip_min=clip_min, clip_max=clip_max, targeted=targeted, ord=ord, rand_init_type=rand_init_type)

