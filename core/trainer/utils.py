
import _pickle as pickle
import argparse
import datetime
from contextlib import contextmanager

import numpy as np
import torch


class CosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing LR schedule (used in Carmon et al, 2019).
    """

    def __init__(self, optimizer, max_lr, epochs, last_epoch=-1):
        self.max_lr = max_lr
        self.epochs = epochs
        self._reset()
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def _reset(self):
        self.current_lr = self.max_lr
        self.current_epoch = 1

    def step(self):
        self.current_lr = self.max_lr * 0.5 * (1 + np.cos((self.current_epoch - 1) / self.epochs * np.pi))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr
        self.current_epoch += 1
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        return self.current_lr


def track_bn_stats(model, track_stats=True):
    """
    If track_stats=False, do not update BN running mean and variance and vice versa.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = track_stats


def set_bn_momentum(model, momentum=1):
    """
    Set the value of momentum for all BN layers.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momentum


def str2bool(v):
    """
    Parse boolean using argument parser.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2float(x):
    """
    Parse float and fractions using argument parser.
    """
    if '/' in x:
        n, d = x.split('/')
        return float(n)/float(d)
    else:
        try:
            return float(x)
        except:
            raise argparse.ArgumentTypeError('Fraction or float value expected.')


def format_time(elapsed):
    """
    Format time for displaying.
    Arguments:
        elapsed: time interval in seconds.
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def seed(seed=1):
    """
    Seed for PyTorch reproducibility.
    Arguments:
        seed (int): Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
def unpickle_data(filename, mode='rb'):
    """
    Read data from pickled file.
    Arguments:
        filename (str): path to the pickled file.
        mode (str): read mode.
    """
    with open(filename, mode) as pkfile:
        data = pickle.load(pkfile)
    return data


def pickle_data(data, filename, mode='wb'):
    """
    Write data to pickled file.
    Arguments:
        data (Any): data to be written.
        filename (str): path to the pickled file.
        mode (str): write mode.
    """
    with open(filename, mode) as pkfile:
         pickle.dump(data, pkfile)


class NumpyToTensor(object):
    """
    Transforms a numpy.ndarray to torch.Tensor.
    """
    def __call__(self, sample): 
        return torch.from_numpy(sample)



class ctx_noparamgrad(object):
    def __init__(self, module):
        self.prev_grad_state = get_param_grad_state(module)
        self.module = module
        set_param_grad_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_param_grad_state(self.module, self.prev_grad_state)
        return False


class ctx_eval(object):
    def __init__(self, module):
        self.prev_training_state = get_module_training_state(module)
        self.module = module
        set_module_training_off(module)

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_module_training_state(self.module, self.prev_training_state)
        return False


@contextmanager
def ctx_noparamgrad_and_eval(module):
    with ctx_noparamgrad(module) as a, ctx_eval(module) as b:
        yield (a, b)


def get_module_training_state(module):
    return {mod: mod.training for mod in module.modules()}


def set_module_training_state(module, training_state):
    for mod in module.modules():
        mod.training = training_state[mod]


def set_module_training_off(module):
    for mod in module.modules():
        mod.training = False


def get_param_grad_state(module):
    return {param: param.requires_grad for param in module.parameters()}


def set_param_grad_state(module, grad_state):
    for param in module.parameters():
        param.requires_grad = grad_state[param]


def set_param_grad_off(module):
    for param in module.parameters():
        param.requires_grad = False