import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from core.utils import SmoothCrossEntropyLoss

softmax = torch.nn.Softmax(1)
crossentropy = nn.CrossEntropyLoss()
from core.metrics import accuracy


def loss_discriminator(opt_D, netD, netG, netC, netA,
           x_l, label, x_u, z_rand, netC_T=None,
           x_u_d=None, unsup_fraction_for_d=0.9, bcr=False
           ):
    if not bcr:
        return loss_hinge_D(opt_D=opt_D, netD=netD, netG=netG, netC=netC, netA=netA, netC_T=netC_T,
                        x_l=x_l, label=label, x_u=x_u, z_rand=z_rand,
                        x_u_d=x_u_d, unsup_fraction_for_d=unsup_fraction_for_d)
    else:
        return loss_hinge_D_bcr(opt_D=opt_D, netD=netD, netG=netG, netC=netC, netA=netA, netC_T=netC_T,
                            x_l=x_l, label=label, x_u=x_u, z_rand=z_rand,
                            x_u_d=x_u_d, unsup_fraction_for_d=unsup_fraction_for_d)



def loss_generator(opt_G, netD, netG, netA,
           label, z_rand):
    opt_G.zero_grad()
    x_g = netG(z_rand, label)
    loss = -torch.mean(netD(x_g, label))
    loss.backward()
    return loss


def get_pert_(model, x_nat, y, step_size=2/255, epsilon=8/255, perturb_steps=10, criterion='ce', nat_logits=None):
    # model.eval()
    x_adv = x_nat.detach() + 0.001 * torch.randn(x_nat.shape).cuda().detach()
    if criterion == 'kl':
        criterion_kl = nn.KLDivLoss(reduction='sum')
    else:
        criterion_ce = SmoothCrossEntropyLoss(reduction='mean')

    if nat_logits is not None:
        p_natural = nat_logits
    else:
        p_natural = softmax(model(x_nat))

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            if criterion == 'kl':
                loss_ = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            else:
                loss_ = criterion_ce(model(x_adv), y)
        grad = torch.autograd.grad(loss_, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_nat - epsilon), x_nat + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        torch.cuda.empty_cache()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # model.train()
    return x_adv



def puat_loss(opt, netD, netG, netC, netC_T, netA,
              x_l, label, x_u, z_rand,
              beta=5.0, beta1=0.03, beta2=1.0,
              adv_ramp = 1.0,
              label_smoothing = 0.,
              consistency_cost = 0.,
              consistency_unsup_frac = 0.9
              ):
    criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    opt.zero_grad()
    zero = torch.tensor(0)

    if adv_ramp > 0:
        # logits_u = netC(x_u)
        # with torch.no_grad():
        #     d_fake_c = netD(x_u)
        # loss_fake = - adv_ramp * beta1 * torch.mean(torch.sum(softmax(logits_u) * d_fake_c, dim=1), dim=0)
        # loss_fake.backward()
        #
        # logits_l = netC(x_l)
        # loss = loss_fake
        # loss_l = loss_consistency = loss_robust_uae = loss_robust_rae = torch.tensor(0)

        # adv_acc, total = 0, 0
        with torch.no_grad():
            x_uae = netG(netA(z_rand, label), label)
            x_g = netG(z_rand, label)
            x_uae = _project(x_uae, x_g, 8 / 255)
        logits_adv = netC(x_uae)
        loss_robust_uae = adv_ramp * beta * criterion_ce(logits_adv, label)
        loss_robust_uae.backward()
        adv_acc_uae = accuracy(label, logits_adv)
        # adv_acc_uae = 0
        # loss_robust_uae = torch.tensor(0)

        logits_l = netC(x_l)
        loss_l = criterion_ce(logits_l, label)
        # loss_l.backward()

        if x_u is not None:
            logits_u = netC(x_u)
        else:
            logits_u = logits_l
            x_u = x_l
        with torch.no_grad():
            d_fake_c = netD(x_u)
            _, y_u = torch.max(netC(x_u), 1)
        loss_fake = - adv_ramp * beta1 * torch.mean(torch.sum(softmax(logits_u) * d_fake_c, dim=1), dim=0)

        with torch.no_grad():
            if x_u is not None:
                num_l = x_l.size(0) - int(x_l.size(0) * consistency_unsup_frac)
                input = torch.cat([x_l[:num_l], x_u[num_l:]], dim=0)
                input_label = torch.cat([label[:num_l], y_u[num_l:]], dim=0)
                prob_C_T = softmax(netC_T(input))
            else:
                prob_C_T = softmax(netC_T(x_l))
        prob_C = softmax(torch.cat([logits_l[:num_l], logits_u[num_l:]]))
        if consistency_cost > 0:
            loss_consistency = consistency_cost * torch.mean((prob_C - prob_C_T) ** 2, dim=[0, 1])
        else:
            loss_consistency = zero
        (loss_l + loss_consistency + loss_fake).backward()

        if beta2 > 0 :
            x_rae = get_pert_(netC, input, input_label, criterion='ce', nat_logits=prob_C)
            logits_adv_rae = netC(x_rae.detach())
            loss_robust_rae = adv_ramp * beta * beta2 * criterion_ce(logits_adv_rae, input_label)
            adv_acc_rae = accuracy(label, logits_adv_rae)
            loss_robust_rae.backward()
        else:
            loss_robust_rae = zero
        loss = loss_l + loss_consistency + loss_robust_rae + loss_robust_uae
        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(label, logits_l.detach()),
                         'adversarial_acc_rae': adv_acc_rae, 'adversarial_acc_uae': adv_acc_uae,
                         }

    else:
        loss_robust_uae = loss_robust_rae = loss_fake = torch.tensor(0)
        logits_l = netC(x_l)
        loss_l = criterion_ce(logits_l, label)
        if consistency_cost > 0:
            if x_u is not None:
                num_l = x_l.size(0) - int(x_l.size(0) * consistency_unsup_frac)
                input = torch.cat([x_l[:num_l], x_u[num_l:]], dim=0)
                with torch.no_grad():
                    prob_C_T = softmax(netC_T(input))
                loss_consistency = consistency_cost * torch.mean((softmax(netC(input)) - prob_C_T) ** 2, dim=[0, 1])
            else:
                with torch.no_grad():
                    prob_C_T = softmax(netC_T(x_l))
                loss_consistency = consistency_cost * torch.mean((softmax(logits_l) - prob_C_T) ** 2, dim=[0, 1])
            loss = loss_l + loss_consistency
        else:
            loss = loss_l
            loss_consistency = torch.tensor(0)
        loss.backward()

        batch_metrics = {'loss': loss.item(), 'clean_acc': accuracy(label, logits_l)}

    loss_dict = {'loss':loss.item(), 'c_sup': loss_l.item(), 'c_con': loss_consistency.item(),
                 'c_fake': loss_fake.item(), 'c_uae': loss_robust_uae.item(), 'c_rae': loss_robust_rae.item()}
    torch.cuda.empty_cache()
    return loss, batch_metrics, loss_dict


def loss_attacker(opt_A, netG, netC, netA,
           label, z_rand, beta, eps=8/255):
    criterion_ce = SmoothCrossEntropyLoss(reduction='mean')
    netC.eval()
    opt_A.zero_grad()
    x_uae = netG(netA(z_rand, label), label)
    with torch.no_grad():
        x_g = netG(z_rand, label)
    x_uae = _project(x_uae, x_g, eps)

    # norm_loss = 100  * torch.mean(torch.relu(torch.abs(x_uae - x_g) - eps))
    norm_loss = torch.tensor(0)
    cla_loss = criterion_ce(netC(x_uae), label)
    loss = - cla_loss + norm_loss
    loss.backward()

    netC.train()
    return loss, cla_loss, norm_loss


def _project(x_adv, x, eps):
    # print("max:{} min:{}".format(torch.max(torch.abs(x_adv-x)), torch.min(torch.abs(x_adv-x))))
    x_adv = torch.min(torch.max(x_adv, x - eps), x + eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv


def loss_hinge_D_bcr(opt_D, netD, netG, netC, netA,
                     x_l, label, x_u, z_rand,
                     netC_T=None, x_u_d=None, unsup_fraction_for_d=0.9,
                     consistency_cost = 100.,
                     ):
    opt_D.zero_grad()

    batch_size = label.size(0)
    bs_u_for_d = int(batch_size * unsup_fraction_for_d)
    bs_l_for_d = batch_size - bs_u_for_d

    with torch.no_grad():
        if x_u_d is None or bs_u_for_d == 0:
            x_real_for_d = x_l
            label_real_for_d = label
        else:
            model = netC_T if netC_T is not None else netC
            x_real_for_d = torch.cat([x_l[: bs_l_for_d], x_u_d[: bs_u_for_d]], 0)
            _, y_l_c = torch.max(model(x_u_d), 1)
            label_real_for_d = torch.cat(
                [label[: bs_l_for_d], y_l_c[: bs_u_for_d]], 0
            )

    d_real = netD(x=x_real_for_d, y=label_real_for_d)
    d_real_2 = netD(x=x_real_for_d, y=label_real_for_d)
    loss_real = torch.mean(torch.relu(1.0 - d_real))
    loss_real_cons = consistency_cost * torch.mean((d_real_2 - d_real) ** 2, dim=[0, 1])
    (loss_real+loss_real_cons).backward()

    with torch.no_grad():
        x_g = netG(z_rand, label).detach()
        y_g = label
        logits_c = netC(x_u).detach()
        _, y_c = torch.max(logits_c, 1)

    d_fake_g = netD(x=x_g, y=y_g)
    d_fake_g_2 = netD(x=x_g, y=y_g)
    loss_fake_g = 0.5 * torch.mean(torch.relu(1.0 + d_fake_g))
    loss_fake_cons = 0.5 * consistency_cost * torch.mean((d_fake_g_2 - d_fake_g) ** 2, dim=[0, 1])
    (loss_fake_g+loss_fake_cons).backward()

    d_fake_c = netD(x_u)
    loss_fake_c = 0.5 * torch.mean(
        torch.sum(torch.relu(1.0 + d_fake_c) * softmax(logits_c), dim=1)
    )
    loss_fake_c.backward()
    loss_fake = loss_fake_g + loss_fake_c
    loss_bcr = loss_fake_cons + loss_real_cons

    return (
        loss_real + loss_fake,
        d_real.mean(),
        d_fake_c.mean(),
        d_fake_g.mean(),
        loss_bcr,
    )


def loss_hinge_D(opt_D, netD, netG, netC, netA,
                 x_l, label, x_u, z_rand,
                 netC_T=None, x_u_d=None, unsup_fraction_for_d=0.9,
                 ):
    opt_D.zero_grad()
    netC.eval()

    batch_size = label.size(0)
    bs_u_for_d = int(batch_size * unsup_fraction_for_d)
    bs_l_for_d = batch_size - bs_u_for_d

    with torch.no_grad():
        if x_u_d is None or bs_u_for_d == 0:
            x_real_for_d = x_l
            label_real_for_d = label
        else:
            model = netC_T if netC_T is not None else netC
            x_real_for_d = torch.cat([x_l[: bs_l_for_d], x_u_d[: bs_u_for_d]], 0)
            _, y_l_c = torch.max(model(x_u_d), 1)
            label_real_for_d = torch.cat(
                [label[: bs_l_for_d], y_l_c[: bs_u_for_d]], 0
            )
    d_real = netD(x=x_real_for_d, y=label_real_for_d)
    loss_real = torch.mean(torch.relu(1.0 - d_real))
    loss_real.backward()

    with torch.no_grad():
        if x_u is None:
            x_u = x_l
        x_g = netG(z_rand, label)
        y_g = label
        logits_c = netC(x_u)

    d_fake_g = netD(x=x_g, y=y_g)
    loss_fake_g = torch.mean(torch.relu(1.0 + d_fake_g))
    d_fake_c = netD(x_u)
    loss_fake_c = torch.mean(
        torch.sum(torch.relu(1.0 + d_fake_c) * softmax(logits_c), dim=1)
    )
    loss_fake = 0.5 * loss_fake_g + 0.5 * loss_fake_c
    loss_fake.backward()
    netC.train()

    return (
        loss_real + loss_fake ,
        d_real.mean(),
        d_fake_c.mean(),
        d_fake_g.mean(),
        torch.tensor(0),
    )

