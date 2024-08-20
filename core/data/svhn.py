import torch

import torchvision
import torchvision.transforms as transforms
from .data_util import MultiDataTransform
from .dataset import SemiSupervisedDataset, InstanceSampleDataset, PseudoDataset
import re

DATA_DESC = {
    'data': 'svhn',
    'classes': ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
    'num_classes': 10,
    'mean': [0.4914, 0.4822, 0.4465], 
    'std': [0.2023, 0.1994, 0.2010],
}

class SemiSupervisedSVHN(SemiSupervisedDataset):
    """
    A dataset with auxiliary pseudo-labeled data for SVHN.
    """
    def load_base_dataset(self, train=False, **kwargs):
        assert self.base_dataset == 'svhn', 'Only semi-supervised svhn is supported. Please use correct dataset!'
        self.dataset = torchvision.datasets.SVHN(split='train' if train else 'test', **kwargs)
        self.num_classes = DATA_DESC['num_classes']
        self.mean_std = (DATA_DESC['mean'], DATA_DESC['std'])


def load_svhn(data_dir, logger, use_augmentation='none', use_consistency=False,
                  take_amount=1000, aux_take_amount=None, take_amount_seed = 1,
                  add_aux_labels=False, validation=False, pseudo_label_model=None,
                  aux_data_filename=None
                  ):
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = SemiSupervisedSVHN(base_dataset='svhn', root=data_dir, train=True, download=True,
                                            transform=train_transform,
                                            take_amount=take_amount,
                                            take_amount_seed=take_amount_seed,
                                            aux_take_amount=aux_take_amount,
                                            validation=validation,
                                            aux_data_filename=aux_data_filename,
                                            add_aux_labels=add_aux_labels,
                                            pseudo_label_model=pseudo_label_model,
                                            logger=logger
                                          )
    test_dataset = SemiSupervisedSVHN(base_dataset='svhn', root=data_dir, train=False, download=True,
                                         transform=test_transform, logger=logger)
    if validation:
        val_dataset = torchvision.datasets.SVHN(root=data_dir, split='train', download=True, transform=test_transform)
        val_dataset = torch.utils.data.Subset(val_dataset, train_dataset.val_indices)
        return train_dataset, test_dataset, val_dataset

    return train_dataset, test_dataset, None
