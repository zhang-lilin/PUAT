import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from core.methods.utils import sigmoid_rampup, ema_update, update_bn, smooth_cross_entropy, set_bn_momentum, \
    track_bn_stats

softmax = torch.nn.Softmax(1)
from core.metrics import accuracy


class PUAT(nn.Module):

    def __init__(self, num_batches, num_epochs, num_classes, data_itr, args, logger,
                 net_D, opt_D, net_G, opt_G, net_A, opt_A,
                 step_size=0.003, epsilon=0.031, perturb_steps=10, beta=10.0,
                 attack='linf-pgd', label_smoothing=0., verbose=False):
        super(PUAT, self).__init__()
        self.params = args
        self.net_D, self.opt_D = net_D, opt_D
        self.net_G, self.opt_G = net_G, opt_G
        self.net_A, self.opt_A = net_A, opt_A
        self.logger = logger
        self.verbose = verbose

        self.num_batches = num_batches
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.current_step = 0
        self.warmup_steps = 0.025 * num_epochs * num_batches
        self.consistency_ramp_up = self.params.consistency_ramp_up * num_batches + 1
        self.consistency_cost = self.params.consistency_cost
        
        self.tau = self.params.tau
        self.tau_after = self.params.tau_after

        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.attack = attack

        self.criterion_ce = smooth_cross_entropy(smoothing=label_smoothing)
        self.criterion_kl = nn.KLDivLoss(reduction='batchmean')

        self.n_iter_d = 5
        self.n_iter_a = 1
        self.data_itr = data_itr


    def forward(self, model, wa_model, optimizer, input, target):
        device, bs = target.device, target.size(0)
        self.current_step = self.current_step + 1
        if self.current_step == 1:
            # make BN running mean and variance init same as Haiku
            set_bn_momentum(model, momentum=1.0)
        elif self.current_step == 2:
            set_bn_momentum(model, momentum=0.01)
        epoch = self.current_step // self.num_batches

        if epoch >= self.params.gan_start:

            for _ in range(self.n_iter_d):
                x, y = self.data_itr.__next__()
                x_l, y_l, x_u = self.split_data(x, y)
                x, y = self.data_itr.__next__()
                _, _, x_u_d = self.split_data(x, y)
                del x, y

                sample_z = torch.randn(bs, self.params.g_z_dim).to(device)
                loss_D, dreal, dfake_c, dfake_g = self.update_discriminator(
                    net_C=model, x_l=x_l, label=y_l, x_u=x_u, z_rand=sample_z, x_u_d=x_u_d,
                    unsup_fraction_for_d=self.params.mix_frac,)

            sample_z = torch.randn(bs, self.params.g_z_dim).to(device)
            loss_G = self.update_generator(net_C=model, label=y_l, z_rand=sample_z)
            if self.verbose:
                if self.current_step % self.num_batches == 0:
                    images = self.eval_generator(device)
                    self.logger.save_images(images, name="img{:03d}".format(epoch), class_name='img', nrow=10)
                    self.plot_loss_gan()
                self.logger.add("training_d", "loss", loss_D.item(), self.current_step)
                self.logger.add("training_d", "dreal", dreal.item(), self.current_step)
                self.logger.add("training_d", "dfake_c", dfake_c.item(), self.current_step)
                self.logger.add("training_d", "dfake_g", dfake_g.item(), self.current_step)
                self.logger.add("training_g", "loss", loss_G.item(), self.current_step)

            if epoch >= self.params.adv_ramp_start:
                for _ in range(self.n_iter_a):
                    loss_A = self.update_attacker(net_C=model, label=y_l, z_rand=sample_z, eps=self.epsilon)
                    if self.verbose:
                        self.logger.add("training_a", "loss", loss_A.item(), self.current_step)

        x_l, y_l, x_u = self.split_data(input, target)
        adv_ramp = sigmoid_rampup(self.current_step, self.params.adv_ramp_start * self.num_batches + 1,
                                  self.params.adv_ramp_end * self.num_batches + 1)
        con_ramp = sigmoid_rampup(self.current_step, 1, self.consistency_ramp_up)
        consistency_cost = self.consistency_cost * con_ramp

        optimizer.zero_grad()
        zero = torch.tensor(0)

        if adv_ramp > 0:
            # sample_z = torch.randn(bs, self.params.g_z_dim).to(device)
            with torch.no_grad():
                x_uae = self.net_G(self.net_A(sample_z, y_l), y_l)
                x_g = self.net_G(sample_z, y_l)
                x_uae = _project(x_uae, x_g, 8 / 255)
            logits_adv = model(x_uae)
            loss_robust_uae = adv_ramp * self.params.lamb * self.criterion_ce(logits_adv, y_l)
            loss_robust_uae.backward()
            adv_acc_uae = accuracy(y_l, logits_adv)

            logits_l = model(x_l)
            loss_l = self.criterion_ce(logits_l, y_l)

            if x_u is not None:
                logits_u = model(x_u)
            else:
                logits_u = logits_l
                x_u = x_l
            with torch.no_grad():
                d_fake_c = self.net_D(x_u)
                _, y_u = torch.max(model(x_u), 1)
            loss_fake = - adv_ramp * self.params.gamma * torch.mean(torch.sum(softmax(logits_u) * d_fake_c, dim=1), dim=0)

            with torch.no_grad():
                if x_u is not None:
                    num_l = x_l.size(0) - int(x_l.size(0) * self.params.mix_frac)
                    input = torch.cat([x_l[:num_l], x_u[num_l:]], dim=0)
                    input_label = torch.cat([y_l[:num_l], y_u[num_l:]], dim=0)
                    prob_C_T = softmax(wa_model(input))
                else:
                    prob_C_T = softmax(wa_model(x_l))
            prob_C = softmax(torch.cat([logits_l[:num_l], logits_u[num_l:]]))
            if consistency_cost > 0:
                loss_consistency = consistency_cost * torch.mean((prob_C - prob_C_T) ** 2, dim=[0, 1])
            else:
                loss_consistency = zero
            (loss_l + loss_consistency + loss_fake).backward()

            if self.params.beta2 > 0:
                x_rae = self.get_adversarial_examples(model, input, input_label, logits_natural=prob_C)
                logits_adv_rae = model(x_rae.detach())
                loss_robust_rae = adv_ramp * self.params.beta * self.criterion_ce(logits_adv_rae, input_label)
                adv_acc_rae = accuracy(y_l, logits_adv_rae)
                loss_robust_rae.backward()
            else:
                loss_robust_rae = zero
            loss = loss_l + loss_consistency + loss_robust_rae + loss_robust_uae
            batch_metrics = {
                'loss': loss.item(),
                'clean_acc': accuracy(y_l, logits_l.detach()),
                'adversarial_acc_rae': adv_acc_rae,
                'adversarial_acc_uae': adv_acc_uae,
            }

        else:
            loss_robust_uae = loss_robust_rae = loss_fake = torch.tensor(0)
            logits_l = model(x_l)
            loss_l = self.criterion_ce(logits_l, y_l)
            if consistency_cost > 0:
                if x_u is not None:
                    num_l = x_l.size(0) - int(x_l.size(0) * self.params.mix_frac)
                    input = torch.cat([x_l[:num_l], x_u[num_l:]], dim=0)
                    with torch.no_grad():
                        prob_C_T = softmax(wa_model(input))
                    loss_consistency = consistency_cost * torch.mean((softmax(model(input)) - prob_C_T) ** 2, dim=[0, 1])
                else:
                    with torch.no_grad():
                        prob_C_T = softmax(wa_model(x_l))
                    loss_consistency = consistency_cost * torch.mean((softmax(logits_l) - prob_C_T) ** 2, dim=[0, 1])
                loss = loss_l + loss_consistency
            else:
                loss = loss_l
                loss_consistency = torch.tensor(0)
            loss.backward()

            batch_metrics = {
                'loss': loss.item(),
                'clean_acc': accuracy(y_l, logits_l)
            }
        if self.verbose:
            self.logger.add("training", "c_sup", loss_l.item(), self.current_step)
            self.logger.add("training", "c_con", loss_consistency.item(), self.current_step)
            self.logger.add("training", "c_fake", loss_fake.item(), self.current_step)
            self.logger.add("training", "c_uae", loss_robust_uae.item(), self.current_step)
            self.logger.add("training", "c_rae", loss_robust_rae.item(), self.current_step)
        torch.cuda.empty_cache()
        optimizer.step()
        self.wa_model_update(model=model, wa_model=wa_model)

        return batch_metrics


    def wa_model_update(self, model, wa_model):
        ema_update(
            wa_model=wa_model,
            model=model,
            global_step=self.current_step,
            decay_rate=self.tau if self.current_step <= self.consistency_ramp_up else self.tau_after,
            warmup_steps=self.warmup_steps,
            dynamic_decay=True
        )
        if self.current_step % self.num_batches == 0:
            update_bn(avg_model=wa_model, model=model)


    def get_adversarial_examples(self, model, input, target, logits_natural=None):
        model.train()
        track_bn_stats(model, False)
        device = input.device

        x_adv = input.detach() + torch.FloatTensor(input.shape).uniform_(-self.epsilon, self.epsilon).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = self.criterion_ce(model(x_adv), target)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, input - self.epsilon), input + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        model.train()
        track_bn_stats(model, True)
        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

        return x_adv


    def split_data(self, input, target):
        device = target.device
        idx = torch.tensor(np.array(range(len(target)))).to(device)
        idx_l, idx_u = idx[target != -1], idx[target == -1]
        if len(idx_u) > 0:
            input_unlabel = input[idx_u]
            input_label, target = input[idx_l], target[idx_l]
            assert input_label.size(0) == target.size(0)
            return input_label.to(device), target.to(device), input_unlabel.to(device)
        else:
            return input.to(device), target.to(device), None


    def update_discriminator(self, net_C, x_l, label, x_u, z_rand, x_u_d=None, unsup_fraction_for_d=0.9):
        self.opt_D.zero_grad()
        net_C.eval()

        batch_size = label.size(0)
        bs_u_for_d = int(batch_size * unsup_fraction_for_d)
        bs_l_for_d = batch_size - bs_u_for_d

        with torch.no_grad():
            if x_u_d is None or bs_u_for_d == 0:
                x_real_for_d = x_l
                label_real_for_d = label
            else:
                model = net_C
                x_real_for_d = torch.cat([x_l[: bs_l_for_d], x_u_d[: bs_u_for_d]], 0)
                _, y_l_c = torch.max(model(x_u_d), 1)
                label_real_for_d = torch.cat(
                    [label[: bs_l_for_d], y_l_c[: bs_u_for_d]], 0
                )
        d_real = self.net_D(x=x_real_for_d, y=label_real_for_d)
        loss_real = torch.mean(torch.relu(1.0 - d_real))
        loss_real.backward()

        with torch.no_grad():
            if x_u is None:
                x_u = x_l
            x_g = self.net_G(z_rand, label)
            y_g = label
            logits_c = net_C(x_u)

        d_fake_g = self.net_D(x=x_g, y=y_g)
        loss_fake_g = torch.mean(torch.relu(1.0 + d_fake_g))
        d_fake_c = self.net_D(x_u)
        loss_fake_c = torch.mean(
            torch.sum(torch.relu(1.0 + d_fake_c) * softmax(logits_c), dim=1)
        )
        loss_fake = 0.5 * loss_fake_g + 0.5 * loss_fake_c
        loss_fake.backward()
        self.opt_D.step()
        net_C.train()
        return loss_real + loss_fake, loss_real, loss_fake_c, loss_fake_g


    def update_generator(self, net_C, label, z_rand):
        self.opt_G.zero_grad()
        x_g = self.net_G(z_rand, label)
        loss = -torch.mean(self.net_D(x_g, label))
        loss.backward()
        self.opt_G.step()
        return loss


    def update_attacker(self, net_C, label, z_rand, eps=8/255):
        net_C.eval()
        self.opt_A.zero_grad()
        x_uae = self.net_G(self.net_A(z_rand, label), label)
        with torch.no_grad():
            x_g = self.net_G(z_rand, label)
        x_uae = _project(x_uae, x_g, eps)
        norm_loss = torch.tensor(0)
        cla_loss = self.criterion_ce(net_C(x_uae), label)
        loss = - cla_loss + norm_loss
        loss.backward()
        self.opt_A.step()
        net_C.train()
        return loss


    def eval_generator(self, device, num_per_class=10):
        test_z_ = torch.randn(num_per_class, self.params.g_z_dim).to(device)
        test_z = torch.cat([test_z_ for _ in range(self.num_classes)], 0).to(device)
        self.net_G.eval()
        with torch.no_grad():
            s = torch.Size([num_per_class])
            test_label = torch.cat([torch.full(s, k) for k in range(self.num_classes)], 0).to(device)
            x_fake = self.net_G(test_z, test_label)
        self.net_G.train()
        return x_fake


    def plot_loss_gan(self):
        cats = ['training_d', 'training_g']
        self.logger.plot_loss(cats, 'gan')
        cats = ['training', 'training_a']
        self.logger.plot_loss(cats, 'adv')


def _project(x_adv, x, eps):
    # print("max:{} min:{}".format(torch.max(torch.abs(x_adv-x)), torch.min(torch.abs(x_adv-x))))
    x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv