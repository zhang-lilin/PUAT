import os
import re

import torch

import os.path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

import torchvision
import torchvision.transforms as transforms
from .dataset import SemiSupervisedDataset
from torchvision.datasets.vision import VisionDataset

DATA_DESC = {
    'data': 'tiny-imagenet',
    'classes': tuple(range(0, 10)),
    'num_classes': 10,
    'mean': [0.4802, 0.4481, 0.3975], 
    'std': [0.2302, 0.2265, 0.2262],
}

class TinyImagenet(VisionDataset):

    path = {
        "train": 'train.npz',
        "val": 'val.npz',
        "test": 'test.npz',
    }
    select = [
        34, 194, 193, 161, 21, 30, 124, 44, 94, 76
    ]

    def __init__(
        self,
        root: str = './dataset_data/tiny-imagenet/',
        train = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        num_labels = 10
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        root = self.root
        split = 'train' if train else 'val'
        fpath = os.path.join(root, self.path[split])

        # reading(loading) npz file as array
        loaded_npz = np.load(fpath)
        if num_labels == 200 or num_labels is None:
            # self.data = loaded_npz['image']
            # self.targets = loaded_npz["label"].tolist()

            X = loaded_npz['image']
            Y = loaded_npz["label"].flatten()
            img_list, label_list = [], []
            for j in range(200):
                tempx = X[Y == j]
                tempy = Y[Y == j] * 0 + j
                img_list.append(tempx)
                label_list.append(tempy)
            self.data = np.concatenate(img_list, axis=0)
            self.targets = np.concatenate(label_list, axis=0)

            print(f'Loading tiny-imagenet-200.')
        else:
            X = loaded_npz['image']
            Y = loaded_npz["label"].flatten()
            img_list, label_list = [], []
            for j in range(num_labels):
                tempx = X[Y == self.select[j]]
                tempy = Y[Y == self.select[j]] * 0 + j
                img_list.append(tempx)
                label_list.append(tempy)
            self.data = np.concatenate(img_list, axis=0)
            self.targets = np.concatenate(label_list, axis=0)
            print(f'Loading tiny-imagenet-{num_labels} {len(self.data)}. ')
            print(f'Label transform : ({self.select[:num_labels]}) to ({set(self.targets)}). ')


    def __getitem__(self, index: int) -> Tuple[Any, Any]:

        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)


    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


class SemiSupervisedTinyImagenet(SemiSupervisedDataset):

    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'tiny-imagenet', 'Only semi-supervised tiny-imagenet is supported. Please use correct dataset!'
        self.dataset = TinyImagenet(train=train, **kwargs)
        self.dataset_size = len(self.dataset)


def load_tinyimagenet(data_dir, logger, use_augmentation='none', use_consistency=False,
                      take_amount=1000, aux_take_amount=None, take_amount_seed = 1,
                      add_aux_labels=False, validation=False, pseudo_label_model=None,
                      aux_data_filename=None):
    test_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose(
            [transforms.Resize(32), transforms.RandomCrop(32, padding=2), transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])
    else:
        train_transform = test_transform

    train_dataset = SemiSupervisedTinyImagenet(base_dataset='tiny-imagenet', root=data_dir, train=True,
                                               transform=train_transform,
                                               take_amount=take_amount,
                                               take_amount_seed=take_amount_seed,
                                               add_aux_labels=add_aux_labels,
                                               aux_take_amount=aux_take_amount,
                                               validation=validation,
                                               pseudo_label_model=pseudo_label_model,
                                               aux_data_filename=aux_data_filename,
                                               logger=logger)
    test_dataset = SemiSupervisedTinyImagenet(base_dataset='tiny-imagenet', root=data_dir, train=False,
                                                transform=test_transform, logger=logger)
    if validation:
        val_dataset = TinyImagenet(root=data_dir, train=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, train_dataset.val_indices)
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset, None


