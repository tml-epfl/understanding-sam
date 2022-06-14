import torch
import numpy as np
import math
from utils import clamp, get_random_delta, project_lp


def get_lr_schedule(lr_schedule_type, n_epochs, lr_max):
    if lr_schedule_type == 'cyclic':
        lr_schedule = lambda epoch: np.interp([epoch], [0, n_epochs * 2 // 5, n_epochs], [0, lr_max, 0])[0]
    elif lr_schedule_type in ['piecewise', 'piecewise_10_100']:
        def lr_schedule(t):
            """
            Following the original ResNet paper (+ warmup for resnet34).
            t is the fractional number of epochs that is passed which starts from 0.
            """
            # if 100 epochs in total, then warmup lasts for exactly 2 first epochs
            # if t / n_epochs < 0.02 and model in ['resnet34']:
            #     return lr_max / 10.
            if t / n_epochs < 0.5:
                return lr_max
            elif t / n_epochs < 0.75:
                return lr_max / 10.
            else:
                return lr_max / 100.
    elif lr_schedule_type in 'piecewise_02epochs':
        def lr_schedule(t):
            if t / n_epochs < 0.2:
                return lr_max
            elif t / n_epochs < 0.75:
                return lr_max / 10.
            else:
                return lr_max / 100.
    elif lr_schedule_type in 'piecewise_03epochs':
        def lr_schedule(t):
            if t / n_epochs < 0.3:
                return lr_max
            elif t / n_epochs < 0.75:
                return lr_max / 10.
            else:
                return lr_max / 100.
    elif lr_schedule_type in 'piecewise_04epochs':
        def lr_schedule(t):
            if t / n_epochs < 0.4:
                return lr_max
            elif t / n_epochs < 0.75:
                return lr_max / 10.
            else:
                return lr_max / 100.
    elif lr_schedule_type == 'piecewise_3_9':
        def lr_schedule(t):
            if t / n_epochs < 0.5:
                return lr_max
            elif t / n_epochs < 0.75:
                return lr_max / 3.
            else:
                return lr_max / 9.
    elif lr_schedule_type == 'piecewise_3_inf':
        def lr_schedule(t):
            if t / n_epochs < 0.5:
                return lr_max
            else:
                return 0.0
    elif lr_schedule_type == 'piecewise_2_3_3':
        def lr_schedule(t):
            if t / n_epochs < 0.15:
                return lr_max
            if t / n_epochs < 0.5:
                return lr_max / 2.
            elif t / n_epochs < 0.75:
                return lr_max / 2. / 3.
            else:
                return lr_max / 2. / 3. / 3.
    elif lr_schedule_type == 'piecewise_5_3_3':
        def lr_schedule(t):
            if t / n_epochs < 0.15:
                return lr_max
            if t / n_epochs < 0.5:
                return lr_max / 5.
            elif t / n_epochs < 0.75:
                return lr_max / 5. / 3.
            else:
                return lr_max / 5. / 3. / 3.
    elif lr_schedule_type == 'piecewise_8_3_3':
        def lr_schedule(t):
            if t / n_epochs < 0.15:
                return lr_max
            if t / n_epochs < 0.5:
                return lr_max / 8.
            elif t / n_epochs < 0.75:
                return lr_max / 10. / 3.
            else:
                return lr_max / 10. / 3. / 3.
    elif lr_schedule_type == 'cosine':
        # cosine LR schedule without restarts like in the SAM paper
        # (as in the JAX implementation used in SAM https://flax.readthedocs.io/en/latest/_modules/flax/training/lr_schedule.html#create_cosine_learning_rate_schedule)
        return lambda epoch: lr_max * (0.5 + 0.5*math.cos(math.pi * epoch / n_epochs))
    elif lr_schedule_type == 'inverted_cosine':
        return lambda epoch: lr_max - lr_max * (0.5 + 0.5*math.cos(math.pi * epoch / n_epochs))
    elif lr_schedule_type == 'constant':
        return lambda epoch: lr_max
    else:
        raise ValueError('wrong lr_schedule_type')
    return lr_schedule


def get_delta_pgd(model, scaler, loss_f, X, y, eps, n_steps_pgd_curr, step_size_pgd_curr, args, use_pred_label=False):
    if args.attack_init == 'zero':
        delta = torch.zeros_like(X, requires_grad=True)
    elif args.attack_init == 'random':
        delta = get_random_delta(X.shape, eps, args.at_norm, requires_grad=True)
        if args.dataset != 'gaussians_binary' and args.model != 'linear':
            delta = clamp(X + delta.data, 0, 1) - X
    else:
        raise ValueError('wrong args.attack_init')

    if args.universal_at:  # note: it's not the same as just averaging deltas bc of the normalization / sign step
        delta = delta[0:1, :, :, :]

    for _ in range(n_steps_pgd_curr):
        with torch.cuda.amp.autocast(enabled=model.half_prec):
            logits = model(X + delta)
            y_adv = logits.max(1)[1].data if use_pred_label else y
            loss = loss_f(logits, y_adv)

        grad = torch.autograd.grad(scaler.scale(loss), delta)[0]
        grad = grad.detach() / scaler.get_scale()

        if args.at_norm == 'l2':
            grad_norms = (grad ** 2).sum([1, 2, 3], keepdim=True) ** 0.5
            grad_norms[grad_norms == 0] = np.inf  # to prevent division by zero
            delta_next = delta.data + step_size_pgd_curr * grad / grad_norms  # step of normalized gradient ascent
        elif args.at_norm == 'linf':
            delta_next = delta.data + step_size_pgd_curr * torch.sign(grad)
        else:
            raise ValueError('wrong args.at_norm')
        delta.data = project_lp(delta_next, args.at_norm, eps)
        if args.dataset != 'gaussians_binary' and args.model != 'linear':
            delta.data = clamp(X + delta.data, 0, 1) - X

    return delta.detach()


def warmup_schedule(n_epochs_warmup, iteration, batch_size, n_train_effective, binary=True):
    n_iters_max = n_epochs_warmup * n_train_effective // batch_size
    coeff = min(iteration / n_iters_max if n_iters_max != 0 else 1.0, 1.0)
    if binary:
        coeff = math.floor(coeff)
    return coeff


def perturb_weights(model, add_weight_perturb_scale, mul_weight_perturb_scale, weight_perturb_distr):
    with torch.no_grad():
        weights_delta_dict = {}
        for param in model.parameters():
            if param.requires_grad:
                if weight_perturb_distr == 'gauss':
                    delta_w_add = add_weight_perturb_scale * torch.randn(param.shape).cuda()  # N(0, std)
                    delta_w_mul = 1 + mul_weight_perturb_scale * torch.randn(param.shape).cuda()  # N(1, std)
                elif weight_perturb_distr == 'uniform':
                    delta_w_add = add_weight_perturb_scale * (torch.rand(param.shape).cuda() - 0.5)  # U(-0.5, 0.5)
                    delta_w_mul = 1 + mul_weight_perturb_scale * (torch.rand(param.shape).cuda() - 0.5)  # U(1 - 0.5*scale, 1 + 0.5*scale)
                else:
                    raise ValueError('wrong weight_perturb_distr')
                param_new = delta_w_mul * param.data + delta_w_add
                delta_w = param_new - param.data
                param.add_(delta_w)
                weights_delta_dict[param] = delta_w  # only the ref to the `param.data` is used as the key
    return weights_delta_dict


def set_weights_to_zero(model):
    for param in model.parameters():
        param.data = torch.zeros_like(param)


def subtract_weight_delta(model, weights_delta_dict):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.sub_(weights_delta_dict[param])  # get back to `w` from `w + delta`


def modify_grads_lin_model(model, x, y, eps, args):
    y_plus_minus = 2 * (y - 0.5)
    with torch.no_grad():  # completely override the gradient
        X_reshaped = x.reshape(x.shape[0], -1)
        w = model._model[1].weight
        w_l2_norm = (w ** 2).sum() ** 0.5
        if args.alpha_weights_linear_at:
            exp = torch.exp(-y_plus_minus[:, None] * X_reshaped @ w.T + eps * w_l2_norm)
            alphas = exp / (1 + exp)
            model._model[1].weight.grad = torch.mean(-alphas * y_plus_minus[:, None] * X_reshaped, 0, keepdim=True)
        if args.alpha_weights_linear_at1:
            exp = torch.exp(-y_plus_minus[:, None] * X_reshaped @ w.T)
            alphas = exp / (1 + exp)
            eps_stability = 0.00001
            model._model[1].weight.grad = torch.mean(
                -alphas * y_plus_minus[:, None] * X_reshaped + eps * w / (w_l2_norm + eps_stability), 0, keepdim=True)
        if args.alpha_weights_linear_at2:
            exp = torch.exp(-y_plus_minus[:, None] * X_reshaped @ w.T)
            alphas = exp / (1 + exp)
            eps_stability = 0.00001
            model._model[1].weight.grad = torch.mean(
                -alphas * y_plus_minus[:, None] * X_reshaped + alphas * eps * w / (w_l2_norm + eps_stability), 0,
                keepdim=True)


def change_bn_mode(model, bn_train):
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if bn_train:
                module.train()
            else:
                module.eval()


def moving_average(net1, net2, alpha=0.999):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    with torch.no_grad():
        model.train()
        momenta = {}
        model.apply(reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        for x, _, _, _, _ in loader:
            x = x.cuda(non_blocking=True)
            b = x.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(x)
            n += b

        model.apply(lambda module: _set_momenta(module, momenta))
        model.eval()


