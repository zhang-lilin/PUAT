import torch
import torchvision
import torchvision.transforms as transforms
import re
import numpy as np
from .autoaugment import CIFAR10Policy
from .idbh import IDBH
from RandAugment import RandAugment # pip install git+https://github.com/ildoonet/pytorch-randaugment


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img



class MultiDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform(sample)
        return x1, x2


class _DataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        x1 = self.transform(x)
        x = transforms.ToTensor()(x)
        return x1, x