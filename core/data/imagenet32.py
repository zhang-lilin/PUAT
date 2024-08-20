import os
import torch

import os.path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.vision import VisionDataset
from .dataset import SemiSupervisedDataset

DATA_DESC = {
    'data': 'tiny-imagenet',
    'classes': tuple(range(0, 10)),
    'num_classes': 100,
    'mean': (0.485, 0.456, 0.406),
    'std': (0.229, 0.224, 0.225),
}


class ImageNet32(VisionDataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    def __init__(self, root,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 use_num_of_class_only=DATA_DESC['num_classes']):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.transform = transform
        self.train = train  # training set or valid set

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for i, (file_name, checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, '{}.npz'.format(file_name))
            entry = np.load(file_path)
            self.data.append(entry["data"])
            self.targets.extend(entry["labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if use_num_of_class_only is not None:
            assert (
                isinstance(use_num_of_class_only, int)
                and use_num_of_class_only > 0
                and use_num_of_class_only < 1000
            ), "invalid use_num_of_class_only : {:}".format(use_num_of_class_only)
            new_data, new_targets = [], []
            for I, L in zip(self.data, self.targets):
                if 1 <= L <= use_num_of_class_only:
                    new_data.append(I)
                    new_targets.append(L - 1)
            print(f'Train={train} new target: {set(new_targets)}')
            self.data = new_data
            self.targets = new_targets

    def __repr__(self):
        return "{name}({num} images, {classes} classes)".format(
            name=self.__class__.__name__,
            num=len(self.data),
            classes=len(set(self.targets)),
        )

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.train_list + self.valid_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                print(filename)
                # return False
        return True


class SemiSupervisedImagenet32(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for tiny-imagenets.
    """
    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'imagenet32', 'Only semi-supervised imagenet32 is supported. Please use correct dataset!'
        self.dataset = ImageNet32(train=train, **kwargs)
        self.dataset_size = len(self.dataset)
        self.num_classes = DATA_DESC['num_classes']
        self.mean_std = (DATA_DESC['mean'], DATA_DESC['std'])


def load_imagenet32(data_dir, logger, use_augmentation='none', use_consistency=False,
                       take_amount=10000, aux_take_amount=None, take_amount_seed=1,
                       add_aux_labels=False, validation=False, pseudo_label_model=None,
                       aux_data_filename=None
                       ):
    """
    Returns semisupervised tiny-imagenets train, test datasets and dataloaders (with DDPM Images).
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (base/none): whether to use augmentations for training set.
        aux_take_amount (int): number of semi-supervised examples to use (if None, use all).
        aux_data_filename (str): path to additional data pickle file.
    Returns:
        train dataset, test dataset.
    """
    test_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    if use_augmentation == 'base':
        train_transform = transforms.Compose(
            [transforms.Resize(32), transforms.RandomCrop(32, padding=4),
             transforms.ToTensor()])
    else:
        train_transform = test_transform

    train_dataset = SemiSupervisedImagenet32(base_dataset='imagenet32', root=data_dir, train=True,
                                               transform=train_transform,
                                               take_amount=take_amount,
                                               take_amount_seed=take_amount_seed,
                                               add_aux_labels=add_aux_labels,
                                               aux_take_amount=aux_take_amount,
                                               validation=validation,
                                               pseudo_label_model=pseudo_label_model,
                                               aux_data_filename=aux_data_filename,
                                               logger=logger
                                               )
    test_dataset = SemiSupervisedImagenet32(base_dataset='imagenet32', root=data_dir, train=False,
                                                transform=test_transform, logger=logger
                                              )
    if validation:
        val_dataset = ImageNet32(root=data_dir, train=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, train_dataset.val_indices)
        return train_dataset, test_dataset, val_dataset
    return train_dataset, test_dataset, None


