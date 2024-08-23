import math
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import copy
import torch
import torch.nn as nn

from core.attacks import create_attack
from core.attacks import CWLoss
from core.utils import ctx_noparamgrad_and_eval
from core.utils import Trainer
from core.utils import set_bn_momentum
from core.utils import seed

from .puat import puat_loss, loss_attacker, loss_generator, loss_discriminator
from core.gan_models import get_generator_optimizer, get_attacker_optimizer, get_discriminator_optimizer



class WATrainer(Trainer):

    def __init__(self, info, args):
        super(WATrainer, self).__init__(info, args)

        seed(args.seed)
        self.wa_model = copy.deepcopy(self.model)
        self.eval_attack = create_attack(self.wa_model, CWLoss, args.attack, args.attack_eps, 4 * args.attack_iter,
                                         args.attack_step)
        num_samples = 50000 if 'cifar' in self.params.data else 73257
        num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
        if 'cifar100' in self.params.data :
            self.num_classes = 100
            self.base_dataset = 'cifar100'
        elif 'cifar10' in self.params.data:
            self.num_classes = 10
            self.base_dataset = 'cifar10'
        elif 'svhn' in self.params.data:
            self.num_classes = 10
            self.base_dataset = 'svhn'
            self.eval_attack = torchattacks.AutoAttack(self.wa_model,eps=args.attack_eps, version='standard', n_classes=10)
        elif 'tiny-imagenet' in self.params.data:
            self.num_classes = 10
            self.base_dataset = 'tiny-imagenet'
        elif 'imagenet32' in self.params.data:
            self.num_classes = 10
            self.base_dataset = 'imagenet32'
        print(f'base dataset: {self.base_dataset}')
        self.update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
        self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps

        if self.params.puat:
            self.init_module_for_puat()


    def init_optimizer(self, num_epochs):
        def group_weight(model):
            group_decay = []
            group_no_decay = []
            for n, p in model.named_parameters():
                if 'batchnorm' in n:
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
            return groups

        self.optimizer = torch.optim.SGD(group_weight(self.model), lr=self.params.lr, weight_decay=self.params.weight_decay, momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs)


    def eval(self, dataloader, adversarial=False, verbose=True):
        self.wa_model.eval()
        device = self.device
        if not adversarial:
            acc, total = 0.0, 0
            for data in tqdm(dataloader, desc='Eval : ', disable=not verbose):
                x, y = data
                x, y = x.to(device), y.to(device)
                total += x.size(0)
                with torch.no_grad():
                    out = self.wa_model(x)
                _, predicted = torch.max(out, 1)
                acc += (predicted == y).sum().item()
            acc /= total
        else:
            if self.base_dataset == 'svhn':
                acc, _, _ = self.eval_attack.save(dataloader, save_path=None, return_verbose=True)
                acc = acc / 100
            else:
                acc, total = 0.0, 0
                for data in tqdm(dataloader, desc='Eval : ', disable=not verbose):
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    total += x.size(0)
                    with ctx_noparamgrad_and_eval(self.wa_model):
                        x_adv, _ = self.eval_attack.perturb(x, y)
                    with torch.no_grad():
                        out = self.wa_model(x_adv)
                    _, predicted = torch.max(out, 1)
                    acc += (predicted == y).sum().item()
                acc /= total
        self.wa_model.train()
        return acc


    def save_model_resume(self, path, epoch):
        """
        Save model weights and optimizer.
        """
        torch.save({
            'model_state_dict': self.wa_model.state_dict(),
            'unaveraged_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch
        }, path)


    def load_model_resume(self, path):
        """
        load model weights and optimizer.
        """
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        self.wa_model.load_state_dict(checkpoint['model_state_dict'])
        self.model.load_state_dict(checkpoint['unaveraged_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']




    def init_module_for_puat(self):
        self.netG, self.optG = get_generator_optimizer(self.params, self.base_dataset)
        self.netD, self.optD = get_discriminator_optimizer(self.params, self.base_dataset, aug=self.params.bcr)
        self.netA, self.optA = get_attacker_optimizer(self.params, self.base_dataset)
        self.bs_g = self.params.bs_g



    def train_puat(self, dataloader, epoch=0, adversarial=False, verbose=False, logger=None):

        metrics = pd.DataFrame()
        self.model.train()
        self.update_steps = len(dataloader)
        data_itr = get_itr(dataloader)

        n_iter_d = 5
        n_iter_a = 1
        device = self.device
        for update_iter in tqdm(range(1, self.update_steps + 1), desc='Epoch {}: '.format(epoch), disable=not verbose):
            global_step = (epoch - 1) * self.update_steps + update_iter
            if global_step == 1:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 2:
                set_bn_momentum(self.model, momentum=0.01)

            if adversarial is True:
                if epoch > self.params.gan_start:
                    """Discriminator"""
                    for _ in range(n_iter_d):
                        x, y = data_itr.__next__()
                        x_l, y_l, x_u = self._data(x, y)
                        x, y = data_itr.__next__()
                        _, _, x_u_d = self._data(x, y)
                        del x, y

                        sample_z = torch.randn(self.bs_g, self.params.g_z_dim).to(device)
                        loss_D, dreal, dfake_c, dfake_g, d_bcr = loss_discriminator(
                            opt_D=self.optD,
                            netD=self.netD, netG=self.netG, netC=self.model, netA=self.netA,
                            x_l=x_l, label=y_l, x_u=x_u, z_rand=sample_z, x_u_d=x_u_d,
                            unsup_fraction_for_d=self.params.unsup_fraction_for_d,
                            netC_T=self.wa_model if self.params.wa_model_for_d else None,
                            bcr=self.params.bcr
                        )
                        self.optD.step()

                    """Generator"""
                    sample_z = torch.randn(self.bs_g, self.params.g_z_dim).to(device)
                    loss_G = loss_generator(
                        opt_G=self.optG,
                        netD=self.netD, netG=self.netG, netA=self.netA,
                        label=y_l, z_rand=sample_z
                    )
                    self.optG.step()

                    if logger is not None:
                        logger.add("training_d", "loss", loss_D.item(), global_step)
                        logger.add("training_d", "dreal", dreal.item(), global_step)
                        logger.add("training_d", "dfake_c", dfake_c.item(), global_step)
                        logger.add("training_d", "dfake_g", dfake_g.item(), global_step)
                        logger.add("training_d", "d_bcr", d_bcr.item(), global_step)
                        logger.add("training_g", "loss", loss_G.item(), global_step)

                    if epoch >= self.params.adv_ramp_start:
                    # if True:
                    #     n_iter_a = 1
                        # adv_ramp = sigmoid_rampup(global_step, self.params.adv_ramp_start * self.update_steps + 1,
                        #                           self.params.adv_ramp_end * self.update_steps + 1)
                        # n_iter_a = math.ceil(adv_ramp * self.params.beta)
                        """Attacker"""
                        for _ in range(n_iter_a):
                            sample_z = torch.randn(self.bs_g, self.params.g_z_dim).to(device)
                            loss_A, loss_atk, loss_norm = loss_attacker(opt_A=self.optA,
                                                   netG=self.netG, netA=self.netA, netC=self.model,
                                                   label=y_l, z_rand=sample_z,
                                                   beta=self.params.beta,
                                                   )
                            self.optA.step()
                        if logger is not None:
                            logger.add("training_a", "loss", loss_A.item(), global_step)
                            logger.add("training_a", "loss_atk", loss_atk.item(), global_step)
                            logger.add("training_a", "loss_nrom", loss_norm.item(), global_step)
                            logger.add("training_a", "grad_norm2",
                                       nn.utils.clip_grad_norm_(self.netA.parameters(), math.inf), global_step)
                            logger.add("training_a", "grad_max",
                                       nn.utils.clip_grad_norm_(self.netA.parameters(), math.inf, norm_type=math.inf),
                                       global_step)

                x, y = data_itr.__next__()
                x_l, y_l, x_u = self._data(x, y)
                del x, y
                adv_ramp = sigmoid_rampup(global_step, self.params.adv_ramp_start * self.update_steps + 1,
                                          self.params.adv_ramp_end * self.update_steps + 1)

                """Classifier"""
                con_ramp = sigmoid_rampup(global_step, 1,
                                          self.params.consistency_ramp_up * self.update_steps + 1)
                loss, batch_metrics, loss_dict = self.puat_loss(
                    x_l, y_l, x_u, beta=self.params.beta, adv_ramp=adv_ramp, adversarial=adversarial, cons_ramp=con_ramp)
                if logger is not None:
                    logger.add("training", "loss", loss_dict['loss'], global_step)
                    logger.add("training", "c_sup", loss_dict['c_sup'], global_step)
                    logger.add("training", "c_con", loss_dict['c_con'], global_step)
                    logger.add("training", "c_fake", loss_dict['c_fake'], global_step)
                    logger.add("training", "c_uae", loss_dict['c_uae'], global_step)
                    logger.add("training", "c_rae", loss_dict['c_rae'], global_step)

            else:
                x, y = data_itr.__next__()
                x_l, y_l, x_u = self._data(x, y)
                del x, y
                """Classifier"""
                con_ramp = sigmoid_rampup(global_step, 1,
                                          self.params.consistency_ramp_up * self.update_steps + 1)
                loss, batch_metrics, loss_dict = self.puat_loss(x_l, y_l, x_u, beta=0., adv_ramp=0., adversarial=adversarial, cons_ramp=con_ramp)

                if logger is not None:
                    logger.add("training", "loss", loss_dict['loss'], global_step)
                    logger.add("training", "c_sup", loss_dict['c_sup'], global_step)
                    logger.add("training", "c_con", loss_dict['c_con'], global_step)

            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()

            ema_update(self.wa_model, self.model, global_step,
                       decay_rate=self.params.tau if epoch <= self.params.consistency_ramp_up else self.params.tau_after,
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)

        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()

        update_bn(self.wa_model, self.model)
        logger.log_info(global_step, ["training_d", "training_g", "training_a", "training"])
        return dict(metrics.mean())


    def _data(self, x, y):
        device = self.device
        idx = np.array(range(len(y)))
        idx_l, idx_u = idx[y != -1], idx[y == -1]
        if len(idx_u) > 0:
            x_u = x[idx_u]
            x_l, y_l = x[idx_l], y[idx_l]
            return x_l.to(device), y_l.to(device), x_u.to(device)
        else:
            return x.to(device), y.to(device), None


    def train_puat_l(self, dataloader, epoch=0, adversarial=False, verbose=False, logger=None):

        metrics = pd.DataFrame()
        self.model.train()
        self.update_steps = len(dataloader)
        data_itr = get_itr(dataloader)

        n_iter_d = 5
        n_iter_a = 1
        device = self.device
        for update_iter in tqdm(range(1, self.update_steps + 1), desc='Epoch {}: '.format(epoch), disable=not verbose):
            global_step = (epoch - 1) * self.update_steps + update_iter
            if global_step == 1:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 2:
                set_bn_momentum(self.model, momentum=0.01)

            if adversarial is True:
                if epoch > self.params.gan_start:
                    """Discriminator"""
                    for _ in range(n_iter_d):
                        x_l, y_l = data_itr.__next__()
                        x_l, y_l = x_l.to(device), y_l.to(device)
                        x_u = None
                        x_u_d = None

                        sample_z = torch.randn(self.bs_g, self.params.g_z_dim).to(device)
                        loss_D, dreal, dfake_c, dfake_g, d_bcr = loss_discriminator(
                            opt_D=self.optD,
                            netD=self.netD, netG=self.netG, netC=self.model, netA=self.netA,
                            x_l=x_l, label=y_l, x_u=x_u, z_rand=sample_z, x_u_d=x_u_d,
                            unsup_fraction_for_d=self.params.unsup_fraction_for_d,
                            netC_T=self.wa_model if self.params.wa_model_for_d else None,
                            bcr=self.params.bcr
                        )
                        self.optD.step()

                    """Generator"""
                    sample_z = torch.randn(self.bs_g, self.params.g_z_dim).to(device)
                    loss_G = loss_generator(
                        opt_G=self.optG,
                        netD=self.netD, netG=self.netG, netA=self.netA,
                        label=y_l, z_rand=sample_z
                    )
                    self.optG.step()

                    if logger is not None:
                        logger.add("training_d", "loss", loss_D.item(), global_step)
                        logger.add("training_d", "dreal", dreal.item(), global_step)
                        logger.add("training_d", "dfake_c", dfake_c.item(), global_step)
                        logger.add("training_d", "dfake_g", dfake_g.item(), global_step)
                        logger.add("training_d", "d_bcr", d_bcr.item(), global_step)
                        logger.add("training_g", "loss", loss_G.item(), global_step)

                    if epoch >= self.params.adv_ramp_start:
                    # if True:
                    #     n_iter_a = 1
                        # adv_ramp = sigmoid_rampup(global_step, self.params.adv_ramp_start * self.update_steps + 1,
                        #                           self.params.adv_ramp_end * self.update_steps + 1)
                        # n_iter_a = math.ceil(adv_ramp * self.params.beta)
                        """Attacker"""
                        for _ in range(n_iter_a):
                            sample_z = torch.randn(self.bs_g, self.params.g_z_dim).to(device)
                            loss_A, loss_atk, loss_norm = loss_attacker(opt_A=self.optA,
                                                   netG=self.netG, netA=self.netA, netC=self.model,
                                                   label=y_l, z_rand=sample_z,
                                                   beta=self.params.beta,
                                                   )
                            self.optA.step()
                        if logger is not None:
                            logger.add("training_a", "loss", loss_A.item(), global_step)
                            logger.add("training_a", "loss_atk", loss_atk.item(), global_step)
                            logger.add("training_a", "loss_nrom", loss_norm.item(), global_step)
                            # logger.add("training_a", "grad_norm2",
                            #            nn.utils.clip_grad_norm_(self.netA.parameters(), math.inf), global_step)
                            # logger.add("training_a", "grad_max",
                            #            nn.utils.clip_grad_norm_(self.netA.parameters(), math.inf, norm_type=math.inf),
                            #            global_step)

                x_l, y_l = data_itr.__next__()
                x_l, y_l = x_l.to(device), y_l.to(device)
                x_u = None
                adv_ramp = sigmoid_rampup(global_step, self.params.adv_ramp_start * self.update_steps + 1,
                                          self.params.adv_ramp_end * self.update_steps + 1)

                """Classifier"""
                con_ramp = sigmoid_rampup(global_step, 1,
                                          self.params.consistency_ramp_up * self.update_steps + 1)
                loss, batch_metrics, loss_dict = self.puat_loss(
                    x_l, y_l, x_u, beta=self.params.beta, adv_ramp=adv_ramp, adversarial=adversarial, cons_ramp=con_ramp)
                if logger is not None:
                    logger.add("training", "loss", loss_dict['loss'], global_step)
                    logger.add("training", "c_sup", loss_dict['c_sup'], global_step)
                    logger.add("training", "c_con", loss_dict['c_con'], global_step)
                    logger.add("training", "c_fake", loss_dict['c_fake'], global_step)
                    logger.add("training", "c_uae", loss_dict['c_uae'], global_step)
                    logger.add("training", "c_rae", loss_dict['c_rae'], global_step)

            else:
                x_l, y_l = data_itr.__next__()
                x_l, y_l = x_l.to(device), y_l.to(device)
                x_u = None
                """Classifier"""
                con_ramp = sigmoid_rampup(global_step, 1,
                                          self.params.consistency_ramp_up * self.update_steps + 1)
                loss, batch_metrics, loss_dict = self.puat_loss(x_l, y_l, x_u, beta=0., adv_ramp=0., adversarial=adversarial, cons_ramp=con_ramp)

                if logger is not None:
                    logger.add("training", "loss", loss_dict['loss'], global_step)
                    logger.add("training", "c_sup", loss_dict['c_sup'], global_step)
                    logger.add("training", "c_con", loss_dict['c_con'], global_step)

            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()

            ema_update(self.wa_model, self.model, global_step,
                       decay_rate=self.params.tau if epoch <= self.params.consistency_ramp_up else self.params.tau_after,
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            metrics = metrics._append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)

        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()

        update_bn(self.wa_model, self.model)
        logger.log_info(global_step, ["training_d", "training_g", "training_a", "training"])
        return dict(metrics.mean())


    def puat_loss(self, x, y, x_u, beta, adv_ramp, cons_ramp, adversarial):
        device = self.device
        sample_z = torch.randn(self.bs_g, self.params.g_z_dim).to(device)
        return puat_loss(
            opt=self.optimizer,
            netD=self.netD, netG=self.netG, netC=self.model,
            netA=self.netA if adversarial else None,
            x_l=x, label=y, x_u=x_u, z_rand=sample_z,
            beta=beta, beta1=self.params.beta1, beta2=self.params.beta2,
            adv_ramp=adv_ramp,
            label_smoothing=self.params.ls,
            netC_T=self.wa_model,
            consistency_cost=self.params.consistency_cost * cons_ramp,
            consistency_unsup_frac=self.params.consistency_unsup_frac
        )


    def eval_generator(self, num_per_class=10):
        device = self.device
        test_z_ = torch.randn(num_per_class, self.params.g_z_dim).to(device)
        test_z = torch.cat([test_z_ for _ in range(self.num_classes)], 0).to(device)
        self.netG.eval()
        with torch.no_grad():
            s = torch.Size([num_per_class])
            test_label = torch.cat([torch.full(s, k) for k in range(self.num_classes)], 0).to(device)
            x_fake = self.netG(test_z, test_label)
        self.netG.train()
        return x_fake


    def eval_atk_generator(self, num_per_class=10):
        device = self.device
        self.netG.eval()
        self.netA.eval()
        with torch.no_grad():
            test_z_ = torch.randn(num_per_class, self.params.g_z_dim).to(device)
            test_z = torch.cat([test_z_ for _ in range(self.num_classes)], 0).to(device)
            s = torch.Size([num_per_class])
            test_label = torch.cat([torch.full(s, k) for k in range(self.num_classes)], 0).to(device)
            z_a = self.netA(test_z, test_label)
            x_fake = self.netG(z_a, test_label)
            x = self.netG(test_z, test_label)
            x_adv = torch.min(torch.max(x_fake, x - 8/255), x + 8/255)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        self.netG.train()
        self.netA.eval()
        return x_adv, x_fake, x


    def save_model_puat(self, path, epoch):
        """
        Save model weights and optimizer.
        """
        torch.save({
            'netC': self.wa_model.state_dict(),
            'unaveraged_netC': self.model.state_dict(),
            'optimizer_C': self.optimizer.state_dict(),
            'scheduler_C': self.scheduler.state_dict() if self.scheduler else None,

            'netG': self.netG.state_dict(),
            'optimizer_G': self.optG.state_dict(),
            'netD': self.netD.state_dict(),
            'optimizer_D': self.optD.state_dict(),
            'netA': self.netA.state_dict(),
            'optimizer_A': self.optA.state_dict(),

            'epoch': epoch
        }, path)


    def load_model_puat(self, path, scheduler=True):
        """
        load model weights and optimizer.
        """
        checkpoint = torch.load(path)
        self.wa_model.load_state_dict(checkpoint['netC'])
        self.model.load_state_dict(checkpoint['unaveraged_netC'])
        if scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_C'])
            self.optimizer.load_state_dict(checkpoint['optimizer_C'])
            self.netA.load_state_dict(checkpoint['netA'])
            self.optA.load_state_dict(checkpoint['optimizer_A'])
        self.netG.load_state_dict(checkpoint['netG'])
        self.optG.load_state_dict(checkpoint['optimizer_G'])
        self.netD.load_state_dict(checkpoint['netD'])
        self.optD.load_state_dict(checkpoint['optimizer_D'])

        return checkpoint['epoch']



def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor
    
    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked



def get_itr(loader, infinity=True):
    if infinity is True:
        while True:
            for img, labels in loader:
                yield img, labels
    else:
        for img, labels in loader:
            yield img, labels



def sigmoid_rampup(global_step, start_iter, end_iter):
    if global_step < start_iter:
        return 0.
    elif start_iter >= end_iter:
        return 1.
    else:
        rampup_length = end_iter - start_iter
        cur_ramp = global_step - start_iter
        cur_ramp = np.clip(cur_ramp, 0, rampup_length)
        phase = 1.0 - cur_ramp / rampup_length
        return np.exp(-5.0 * phase * phase)



