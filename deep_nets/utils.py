import logging
import torch
from contextlib import contextmanager


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def clamp(X, l, u, cuda=True):
    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)


def configure_logger(model_name, debug):
    logging.basicConfig(format='%(message)s')  # , level=logging.DEBUG)
    logger = logging.getLogger()
    logger.handlers = []  # remove the default logger

    # add a new logger for stdout
    formatter = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if not debug:
        # add a new logger to a log file
        logger.addHandler(logging.FileHandler('logs/{}.log'.format(model_name)))

    return logger


def get_random_delta(shape, eps, at_norm, requires_grad=True):
    delta = torch.zeros(shape).cuda()
    if at_norm == 'l2':  # uniform from the hypercube
        delta.normal_()
        delta /= (delta**2).sum([1, 2, 3], keepdim=True)**0.5
    elif at_norm == 'linf':  # uniform on the sphere
        delta.uniform_(-eps, eps)
    else:
        raise ValueError('wrong at_norm')
    delta.requires_grad = requires_grad
    return delta


def project_lp(img, at_norm, eps):
    if at_norm == 'l2':  # uniform on the sphere
        l2_norms = (img ** 2).sum([1, 2, 3], keepdim=True) ** 0.5
        img_proj = img * torch.min(eps/l2_norms, torch.ones_like(l2_norms))  # if eps>l2_norms => multiply by 1
    elif at_norm == 'linf':  # uniform from the hypercube
        img_proj = clamp(img, -eps, eps)
    else:
        raise ValueError('wrong at_norm')
    return img_proj


def update_metrics(metrics_dict, metrics_values, metrics_names):
    assert len(metrics_values) == len(metrics_names)
    for metric_value, metric_name in zip(metrics_values, metrics_names):
        metrics_dict[metric_name].append(metric_value)
    return metrics_dict


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result


def get_flat_grad(model):
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def zero_grad(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


def eval_f_val_grad(model, f):
    zero_grad(model)

    obj = f(model)  # grads are accumulated
    obj.backward()
    grad_norm = get_flat_grad(model).norm()

    zero_grad(model)
    return obj.detach(), grad_norm.detach()

