import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.attacks import CWLoss
from core.attacks import create_attack
from core.metrics import accuracy
from core.models import create_model

from .context import ctx_noparamgrad_and_eval
from .utils import seed
SCHEDULERS = ['cyclic', 'step', 'cosine', 'cosinew']


class Trainer(object):
    def __init__(self, info, args, device_ids=None):
        super(Trainer, self).__init__()
        device = self.device = args.device
        
        seed(args.seed)
        self.model = create_model(args, info, device)

        self.params = args
        self.criterion = nn.CrossEntropyLoss()
        self.init_optimizer(self.params.num_adv_epochs)
        
        if self.params.pretrained_file is not None:
            self.load_model(os.path.join(self.params.log_dir, self.params.pretrained_file, 'weights-best.pt'))
        
        self.attack = create_attack(self.model, self.criterion, args.attack, args.attack_eps, args.attack_iter, args.attack_step, rand_init_type='uniform')
        self.eval_attack = create_attack(self.model, CWLoss, args.attack, args.attack_eps, 4 * args.attack_iter,
                                         args.attack_step)
        
    
    @staticmethod
    def init_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step):
        """
        Initialize adversary.
        """
        attack = create_attack(model, criterion, attack_type, attack_eps, attack_iter, attack_step, rand_init_type='uniform')
        if attack_type in ['linf-pgd', 'l2-pgd']:
            eval_attack = create_attack(model, criterion, attack_type, attack_eps, 2*attack_iter, attack_step)
        elif attack_type in ['fgsm', 'linf-df']:
            eval_attack = create_attack(model, criterion, 'linf-pgd', 8/255, 20, 2/255)
        elif attack_type in ['fgm', 'l2-df']:
            eval_attack = create_attack(model, criterion, 'l2-pgd', 128/255, 20, 15/255)
        return attack,  eval_attack
    
    
    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and scheduler.
        """
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)
    
        
    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'cyclic':
            num_samples = 50000 if 'cifar10' in self.params.data else 73257
            num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
            update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.25,
                                                                 steps_per_epoch=update_steps, epochs=int(num_epochs))
        elif self.params.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[100, 105])    
        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.025, 
                                                                 total_steps=int(num_epochs))
        else:
            self.scheduler = None

    
    def eval(self, dataloader, adversarial=False, verbose=True):
        acc = 0.0
        total = 0
        self.model.eval()
        # for x, y in dataloader:
        device = self.device
        self.model = self.model.to(device)
        for data in tqdm(dataloader, desc='Eval : ', disable=not verbose):
            x, y = data
            x, y = x.to(device), y.to(device)
            total += x.size(0)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.eval_attack.perturb(x, y)
                with torch.no_grad():
                    out = self.model(x_adv)
            else:
                with torch.no_grad():
                    out = self.model(x)
            _, predicted = torch.max(out, 1)
            acc += (predicted == y).sum().item()
        acc /= total
        self.model.train()
        return acc


    def save_model(self, path, epoch):
        """
        Save model weights.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }, path)

    
    def load_model(self, path, load_opt=True):
        """
        Load model weights.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except:
            self.model = torch.nn.DataParallel(self.model)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return checkpoint['epoch']

