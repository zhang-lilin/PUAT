import numpy as np
import torch


class smooth_cross_entropy(torch.nn.Module):
    """
    Cross entropy loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        super(smooth_cross_entropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target, reduction='mean'):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        return loss


def track_bn_stats(model, track_stats=True):
    """
    If track_stats=False, do not update BN running mean and variance and vice versa.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = track_stats


def set_bn_momentum(model, momentum=1):
    """
    Set the value of momentum for all BN layers.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momentum


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


class CW_log():
    def __init__(self, class_num=10) -> None:
        self.class_num = class_num
        self.reset()

    def update(self, output, output_adv, y):
        pred = output.max(1)[1]
        pred_adv = output_adv.max(1)[1]
        correct = pred == y
        correct_adv = pred_adv == y
        for index, label in enumerate(y):
            self.cw_n[label] += 1
            if correct[index]:
                self.cw_natural[label] += 1
            if correct_adv[index]:
                self.cw_robust[label] += 1

    def result(self):
        return self.cw_natural / self.cw_n, self.cw_robust / self.cw_n

    def reset(self):
        self.cw_n = torch.zeros(self.class_num)
        self.cw_robust = torch.zeros(self.class_num)
        self.cw_natural = torch.zeros(self.class_num)
