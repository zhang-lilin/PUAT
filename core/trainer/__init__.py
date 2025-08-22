from .logger import Logger
from .parser import *
from .utils import *
import copy
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm
from core.attacks import create_attack
from core.models import create_model
from .utils import CosineLR

class Trainer(object):
    """
    Helper class for training a deep neural network.
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """

    def __init__(self, info, args, logger, dataloader):
        super(Trainer, self).__init__()
        device = self.device = args.device

        seed(args.seed)
        self.logger = logger
        self.params = args
        self.info = info

        self.model = create_model(args, info, device, logger)
        self.init_optimizer(self.params.num_epochs)
        self.init_loss(args.method, args, dataloader)
        self.init_scheduler(self.params.num_epochs)

        self.wa_model = None
        if args.tau > 0:
            logger.log('Using WA.')
            self.wa_model = copy.deepcopy(self.model)
        self.init_attack(self.get_model(), self.params.attack)

    def init_attack(self, model, attack_type):
        """
        Initialize adversary.
        """
        criterion = nn.CrossEntropyLoss()
        if attack_type in ['linf-pgd', 'l2-pgd']:
            self.eval_attack = create_attack(model, criterion, attack_type, 8 / 255, 20, 2 / 255)
        if attack_type in ['fgsm', 'linf-df']:
            self.eval_attack = create_attack(model, criterion, 'linf-pgd', 8 / 255, 20, 2 / 255)
        elif attack_type in ['fgm', 'l2-df']:
            self.eval_attack = create_attack(model, criterion, 'l2-pgd', 128 / 255, 20, 15 / 255)

    def init_optimizer(self, num_epochs):
        """
        Initialize optimizer and scheduler.
        """
        if self.params.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr,
                                             weight_decay=self.params.weight_decay,
                                             momentum=0.9, nesterov=self.params.nesterov)
            if self.logger:
                self.logger.log(
                    f"SGD Optimizer: lr-{self.params.lr} momentum-0.9 nesterove-{self.params.nesterov} weight_decay-{self.params.weight_decay}")
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr)

    def init_scheduler(self, num_epochs):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'cyclic':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.25,
                                                                 steps_per_epoch=self.update_steps,
                                                                 epochs=int(num_epochs))

        elif self.params.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **self.params.scheduler_opt)
            if self.logger:
                self.logger.log(f'LR scheduler: step {self.params.scheduler_opt}')

        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
            if self.logger:
                self.logger.log(f'LR scheduler: cosine max_lr-{self.params.lr} epochs-{int(num_epochs)}')

        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=0.05,
                                                                 total_steps=int(num_epochs))
        elif self.params.scheduler == 'none':
            self.scheduler = None

        else:
            raise NotImplementedError(self.params.scheduler)

    def init_loss(self, method, args, dataloader=None):

        data_param = dict(num_batches=args.num_batches, num_epochs=args.num_epochs)
        adv_train_params = dict(
            step_size=args.attack_step,
            epsilon=args.attack_eps,
            perturb_steps=args.attack_iter,
            attack=args.attack, )

        if 'puat' in method:
            from core.gan_models import get_generator_optimizer, get_attacker_optimizer, get_discriminator_optimizer
            net_G, opt_G = get_generator_optimizer(self.params, self.info['data'])
            net_D, opt_D = get_discriminator_optimizer(self.params, self.info['data'])
            net_A, opt_A = get_attacker_optimizer(self.params, self.info['data'])
            nets = dict(
                net_G=net_G, opt_G=opt_G,
                net_D=net_D, opt_D=opt_D,
                net_A=net_A, opt_A=opt_A,
            )
            def get_itr(loader, device, infinity=True):
                if infinity is True:
                    while True:
                        for img, labels in loader:
                            yield img.to(device), labels.to(device)
                else:
                    for img, labels in loader:
                        yield img.to(device), labels.to(device)

            data_itr = get_itr(dataloader, self.device, infinity=True)
            from core.methods.puat import PUAT
            self.loss = PUAT(data_itr=data_itr, args=self.params, logger=self.logger, **nets,
                             num_classes=self.info['num_classes'], num_batches=args.num_batches,
                             num_epochs=args.num_epochs, **adv_train_params, )

        else:
            raise NotImplementedError

        self.loss.to(self.device)

    def train(self, dataloader, epoch, verbose=True):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()
        update_iter = 1
        device = self.device
        for (input, target) in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            global_step = (epoch - 1) * len(dataloader) + update_iter
            target = target.to(device)
            try:
                input = input.to(device)
            except:
                for i in range(len(input)):
                    input[i] = input[i].to(device)

            loss_params = dict(
                model=self.model, wa_model=self.wa_model,
                input=input, target=target, optimizer=self.optimizer,
            )
            batch_metrics = self.loss(**loss_params)
            try:
                metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
            except:
                metrics = metrics._append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
            if verbose:
                for key in batch_metrics:
                    if 'loss' in key:
                        self.logger.add("training", key, batch_metrics[key], global_step)
            update_iter += 1

        self.scheduler.step()
        if verbose:
            self.logger.log_info(global_step, ["training"])

        return dict(metrics.mean())

    def model_parameters(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                yield p

    def eval(self, dataloader, adversarial=False, verbose=True):
        model = self.get_model()
        model.eval()
        acc, total = 0.0, 0
        for data in tqdm(dataloader, desc='Eval : ', disable=not verbose):
            x, y = data
            x, y = x.to(self.device), y.to(self.device)
            total += x.size(0)
            if adversarial:
                with ctx_noparamgrad_and_eval(model):
                    x_adv, _ = self.eval_attack.perturb(x, y)
                with torch.no_grad():
                    out = model(x_adv)
            else:
                with torch.no_grad():
                    out = model(x)
            _, predicted = torch.max(out, 1)
            acc += (predicted == y).sum().item()
        acc /= total
        model.train()

        return acc

    def get_model(self):
        if self.wa_model is None:
            return self.model
        else:
            return self.wa_model

    def save_model(self, path, epoch):
        if self.scheduler is not None:
            scheduler_state_dict = self.scheduler.state_dict()
        else:
            scheduler_state_dict = None

        if self.wa_model is None:
            save_dict = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': scheduler_state_dict,
                'epoch': epoch,
            }

        if 'puat' in self.params.method:
            save_dict = {
                'model_state_dict': self.wa_model.state_dict(),
                'unaveraged_model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': scheduler_state_dict,

                'net_G': self.loss.net_G.state_dict(),
                'optimizer_G': self.loss.opt_G.state_dict(),
                'net_D': self.loss.net_D.state_dict(),
                'optimizer_D': self.loss.opt_D.state_dict(),
                'net_A': self.loss.net_A.state_dict(),
                'optimizer_A': self.loss.opt_A.state_dict(),

                'epoch': epoch
            }

        else:
            save_dict = {
                'model_state_dict': self.wa_model.state_dict(),
                'unaveraged_model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': scheduler_state_dict,
                'epoch': epoch,
            }

        if hasattr(self.loss, 'save_list'):
            for key in self.loss.save_list:
                save_dict[f'loss_opt_{key}'] = getattr(self.loss, key)

        torch.save(save_dict, path)

    def load_model(self, path, load_opt=True):
        """
        load model weights and optimizer.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        else:
            if self.wa_model is not None:
                self.wa_model.load_state_dict(checkpoint['model_state_dict'])
                self.model.load_state_dict(checkpoint['unaveraged_model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])

            if load_opt:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if 'puat' in self.params.method:
                self.loss.net_A.load_state_dict(checkpoint['net_A'])
                self.loss.opt_A.load_state_dict(checkpoint['optimizer_A'])
                self.loss.net_G.load_state_dict(checkpoint['net_G'])
                self.loss.opt_G.load_state_dict(checkpoint['optimizer_G'])
                self.loss.net_D.load_state_dict(checkpoint['net_D'])
                self.loss.opt_D.load_state_dict(checkpoint['optimizer_D'])

            if hasattr(self.loss, 'current_step'):
                self.loss.current_step = checkpoint['epoch'] * self.loss.num_batches

        return checkpoint['epoch']
