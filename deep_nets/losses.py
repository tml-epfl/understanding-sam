import torch
import torch.nn.functional as F


def cross_entropy(reduction=True):
    def loss_f(logits, y):
        loss = F.cross_entropy(logits, y, reduction='mean' if reduction else 'none')
        return loss

    return loss_f


def cross_entropy_with_offset(loss_offset):
    """ Works only for binary classification. """
    def loss_f(logits, y, reduction=True):
        assert max(y).item() <= 1 and min(y).item() >= 0  # i.e. binary labels directly encoded as 0 or 1
        y_plus_minus = 2 * (y - 0.5)
        loss = torch.log(1 + torch.exp(-y_plus_minus * (logits[:, 1] - logits[:, 0]) + loss_offset))
        return loss.mean() if reduction else loss

    return loss_f


def logistic_loss():
    """ Works only for binary classification. Assumes logits have only 1 prediction. """
    def loss_f(logits, y, reduction=True):
        assert max(y).item() <= 1 and min(y).item() >= 0  # i.e. binary labels directly encoded as 0 or 1
        y_plus_minus = 2 * (y - 0.5)
        loss = torch.log(1 + torch.exp(-y_plus_minus * (logits[:, 1] - logits[:, 0])))
        return loss.mean() if reduction else loss

    return loss_f


def generalized_cross_entropy(q):
    def loss_f(logits, y, reduction=True):
        p = F.softmax(logits, dim=1)
        p_y = p[range(p.shape[0]), y]
        loss = 1/q * (1 - p_y**q)
        return loss.mean() if reduction else loss

    return loss_f


def smallest_k_cross_entropy(frac_rm_per_batch):
    def loss_f(logits, y, reduction=True):
        k_keep = y.shape[0] - int(frac_rm_per_batch * y.shape[0])
        loss = F.cross_entropy(logits, y, reduction='none')
        loss = torch.topk(loss, k_keep, largest=False)[0]  # take `k_keep` smallest losses
        return loss.mean() if reduction else loss

    return loss_f


def logistic_loss_der(logits, y):
    """ Works only for binary classification. Assumes logits have only 1 prediction. """
    y_plus_minus = 2 * (y - 0.5)
    der = -y_plus_minus/(1 + torch.exp(y_plus_minus * (logits[:, 1] - logits[:, 0])))
    return der

