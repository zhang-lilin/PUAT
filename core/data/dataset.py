import os
import pickle
import numpy as np
import torch.utils.data
from tqdm import tqdm
import torchvision


VALIDATION_FRACTION = 0.2
select = [
        34, 194, 193, 161, 21, 30, 124, 44, 94, 76
    ]


class SemiSupervisedDataset(torch.utils.data.Dataset):

    def __init__(self, logger, base_dataset='cifar10', take_amount=4000, take_amount_seed=1,
                 add_aux_labels=False, aux_take_amount=None, train=True, validation=False,
                 pseudo_label_model=None, aux_data_filename=None, **kwargs):

        self.base_dataset = base_dataset
        self.load_base_dataset(train, **kwargs)
        if 'svhn' in base_dataset:
            self.dataset.labels = self.dataset.labels.tolist()
        
        self.train = train
        self.origen_len = self.__len__()

        if self.train:
            if take_amount is not None:
                assert take_amount <= self.__len__()
                # frac = str(VALIDATION_FRACTION).replace('.', 'p')
                split_path = os.path.join(kwargs['root'], f'split-{take_amount}-seed-{take_amount_seed}.npz')

                if os.path.exists(split_path):
                    split_info = np.load(split_path)
                    self.unsup_indices = split_info['unsup_indices']
                    self.sup_indices = split_info['sup_indices']

                    if validation:
                        sup_indices = self.sup_indices
                        take_amount_per_label = (take_amount // self.num_classes)
                        val_indices, take_amount_val = [], min(int(take_amount * VALIDATION_FRACTION), 2000)
                        sup_indices_train = []
                        for label in range(self.num_classes):
                            tind = sup_indices[take_amount_per_label * label: take_amount_per_label * (label + 1)]
                            val_indices.extend(tind[: take_amount_val // self.num_classes])
                            sup_indices_train.extend(tind[take_amount_val // self.num_classes :])
                        self.val_indices = val_indices
                        self.sup_indices = sup_indices_train
                    else:
                        self.val_indices = []
                else:
                    sup_indices, unsup_indices = [], []
                    indexs = np.array(range(self.__len__()))
                    generator = np.random.default_rng(take_amount_seed)
                    self.num_classes = len(set(self.dataset.targets))
                    
                    assert take_amount // self.num_classes > 0
                    take_amount_per_label = (take_amount // self.num_classes)
                    for label in range(self.num_classes):
                        tind = list(indexs[np.array(list(self.dataset.targets)) == label])
                        generator.shuffle(tind)
                        sup_indices.extend(tind[: take_amount_per_label])
                        unsup_indices.extend(tind[take_amount_per_label:])
                    self.unsup_indices = unsup_indices
                    self.sup_indices = sup_indices
                    np.savez(split_path, unsup_indices=unsup_indices, sup_indices=sup_indices)

                    if validation:
                        val_indices, take_amount_val = [], int(take_amount * VALIDATION_FRACTION)
                        sup_indices_train = []
                        for label in range(self.num_classes):
                            tind = sup_indices[take_amount_per_label * label: take_amount_per_label * (label + 1)]
                            val_indices.extend(tind[: take_amount_val // self.num_classes])
                            sup_indices_train.extend(tind[take_amount_val // self.num_classes:])
                        self.val_indices = val_indices
                        self.sup_indices = sup_indices_train
                    else:
                        self.val_indices = []

            else:
                self.unsup_indices = []
                data_amount = len(self.targets)
                if validation:
                    take_amount_val = min(int(data_amount * VALIDATION_FRACTION), 2000)
                    self.val_indices, self.sup_indices = range(0, take_amount_val), range(take_amount_val, data_amount)
                else:
                    self.sup_indices = range(0, data_amount)
                    self.val_indices = []

            if aux_data_filename is None:
                # real labels
                if add_aux_labels:
                    pass
                # pseudo labels
                elif pseudo_label_model is not None:
                    batch_size, data_size = 1024, len(self.data)
                    batch_num = int(np.ceil(len(self.data)/batch_size))

                    acc, total = 0, 0
                    for i in tqdm(range(batch_num), desc='Pseudo: '):
                        data = None
                        end = min((i+1)*batch_size, data_size)
                        for k in range(i*batch_size,end):
                            data_k = self.data[k]
                            if 'svhn' in base_dataset:
                                data_k = data_k.transpose(1, 2, 0)
                            elif 'tiny-imagenet' in base_dataset:
                                data_k = torchvision.transforms.ToPILImage()(data_k)
                                data_k = torchvision.transforms.Resize(32)(data_k)
                            data_k = torchvision.transforms.ToTensor()(data_k)
                            data = data_k.unsqueeze(0) if data is None else torch.cat([data, data_k.unsqueeze(0)], 0)

                        with torch.no_grad():
                            # print(data.size(0), data.size(1), data.size(2), data.size(3))
                            _, label = torch.max(pseudo_label_model(data.cuda()).detach(), 1)

                        for k in range(i*batch_size,end):
                            if k in self.unsup_indices:
                                total += 1
                                if self.targets[k] == int(label[k % batch_size]):
                                    acc += 1
                                self.targets[k] = int(label[k % batch_size])
                    logger.log(f'Pseudo total:{total} acc:{acc * 100 / total}%.')
                # no labels
                else:
                    for i in self.unsup_indices:
                        self.targets[i] = -1
            else:
                aux_path = aux_data_filename
                self.unsup_indices = []

                logger.log('Loading auxiliary data from %s' % aux_path)
                if os.path.splitext(aux_path)[1] == '.pickle':
                    # for data from Carmon et al, 2019.
                    with open(aux_path, 'rb') as f:
                        aux = pickle.load(f)
                    aux_data = aux['data']
                    aux_targets = aux['extrapolated_targets']
                else:
                    # for data from Rebuffi et al, 2021.
                    aux = np.load(aux_path)
                    aux_data = aux['image']
                    print(aux_data.shape)
                    aux_targets = aux['label']

                if aux_take_amount is not None:
                    rng_state = np.random.get_state()
                    np.random.seed(take_amount_seed)
                    take_inds = np.random.choice(len(aux_data), aux_take_amount, replace=False)
                    np.random.set_state(rng_state)
                    aux_data = aux_data[take_inds]
                    aux_targets = aux_targets[take_inds]

                if 'tiny-imagenet' in base_dataset:
                    img_list, label_list = [], []
                    for j in range(10):
                        tempx = aux_data[aux_targets == self.select[j]]
                        tempy = aux_targets[aux_targets == self.select[j]] * 0 + j
                        img_list.append(tempx)
                        label_list.append(tempy)
                    aux_data = np.concatenate(img_list, axis=0)
                    aux_targets = np.concatenate(label_list, axis=0)
                    assert len(aux_data) == len(aux_targets)
                    logger.log(f'Tiny-ImageNet transform in aux data. amount {len(aux_data)}')
                elif 'svhn' in base_dataset:
                    logger.log('SVHN transform in aux data.')
                    aux_data = aux_data.transpose(0, 3, 1, 2)

                orig_len = len(self.data)
                self.data = np.concatenate((self.data, aux_data), axis=0)
                self.unsup_indices.extend(range(orig_len, orig_len + len(aux_data)))

                # self.targets = self.targets.tolist()
                if add_aux_labels:
                    self.targets.extend(aux_targets)
                # pseudo labels
                if pseudo_label_model is not None:
                    batch_size, data_size = 1024, len(aux_data)
                    import math
                    batch_num = math.ceil(data_size / batch_size)

                    acc, total = 0, 0
                    for i in tqdm(range(batch_num), desc='Pseudo: '):
                        data = None
                        last = min((i + 1) * batch_size, data_size)
                        for k in range(i * batch_size, last):
                            data_k = aux_data[k]
                            if 'svhn' in base_dataset:
                                data_k = data_k.transpose(1, 2, 0)
                            elif 'tiny-imagenet' in base_dataset:
                                data_k = torchvision.transforms.ToPILImage()(data_k)
                                data_k = torchvision.transforms.Resize(32)(data_k)
                            data_k = torchvision.transforms.ToTensor()(data_k)
                            data = data_k.unsqueeze(0) if data is None else torch.cat([data, data_k.unsqueeze(0)], 0)

                        with torch.no_grad():
                            _, label = torch.max(pseudo_label_model(data.cuda()).detach(), 1)

                        for k in range(i * batch_size, last):
                            total += 1
                            if aux_targets[k] == int(label[k % batch_size]):
                                acc += 1
                            aux_targets[k] = int(label[k % batch_size])
                        # print(f'Pseudo total:{total} acc:{acc * 100 / total}.')
                    logger.log(f'Pseudo total:{total} acc:{acc * 100 / total}.')
                    try:
                        self.targets.extend(aux_targets)
                    except:
                        self.dataset.targets = self.dataset.targets.tolist()
                        self.targets.extend(aux_targets)
                        assert len(self.targets) == len(self.data)
                # no labels
                else:
                    self.targets.extend([-1] * len(aux_data))


            if aux_take_amount is not None:
                assert aux_take_amount >= 0
                assert len(self.unsup_indices) >= aux_take_amount, "No enough unlabeled data exits."
                rng_state = np.random.get_state()
                np.random.seed(take_amount_seed)
                take_inds = np.random.choice(self.unsup_indices, aux_take_amount, replace=False)
                np.random.set_state(rng_state)
                self.unsup_indices = take_inds
                split_path = os.path.join(kwargs['root'], f'split-unlabel-{aux_take_amount}-seed-{take_amount_seed}.npz')
                np.savez(split_path, unsup_indices=self.unsup_indices)

            logger.log(f"Training Dataset:  label:{len(self.sup_indices)}  unlabel:{len(self.unsup_indices)}  validation:{len(self.val_indices)}")

        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

        self.dataset.labels = self.targets



    def load_base_dataset(self, **kwargs):
        raise NotImplementedError()

    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        try:
            return self.dataset.targets
        except:
            return self.dataset.labels

    @targets.setter
    def targets(self, value):
        try:
            self.dataset.labels = value
        except:
            self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def semi_info(self):
        if self.train:
            return len(self.sup_indices), len(self.unsup_indices), len(self.val_indices)
        else:
            return len(self.sup_indices)

    def __getitem__(self, item):
        self.dataset.labels = self.targets
        return self.dataset[item]
