import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import copy
import utils
import utils_eval
import utils_train
import data
import models
import losses
from collections import defaultdict
from datetime import datetime
from models import forward_pass_rlat
from sam import SAM, perturb_weights_sam


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset', default='cifar10', choices=data.datasets_dict.keys(), type=str)
    parser.add_argument('--model', default='resnet18', choices=['vgg16', 'resnet18', 'resnet18_gn', 'resnet_tiny', 'resnet_tiny_gn', 'resnet34', 'resnet34preact', 'resnet34_gn', 'wrn28', 'lenet', 'cnn', 'fc', 'linear'], type=str)
    parser.add_argument('--epochs', default=100, type=int, help='100 epochs is standard with batch_size=128')
    parser.add_argument('--lr_schedule', default='piecewise', choices=['cyclic', 'piecewise', 'cosine', 'constant', 'piecewise_02epochs', 'piecewise_03epochs', 'piecewise_04epochs'])
    parser.add_argument('--ln_schedule', default='constant', choices=['constant', 'inverted_cosine', 'piecewise_10_100', 'piecewise_3_9', 'piecewise_3_inf', 'piecewise_2_3_3', 'piecewise_5_3_3', 'piecewise_8_3_3'])
    parser.add_argument('--lr_max', default=0.1, type=float, help='')
    parser.add_argument('--p_label_noise', default=0.0, type=float, help='Fraction of flipped labels in the training data.')
    parser.add_argument('--noise_type', default='sym', type=str, choices=['sym', 'asym'], help='Noise type: symmetric or asymmetric')
    parser.add_argument('--attack', default='none', type=str, choices=['fgsm', 'fgsmpp', 'pgd', 'rlat', 'random_noise', 'none'])
    parser.add_argument('--at_norm', default='linf', type=str, choices=['l2', 'linf'])
    parser.add_argument('--universal_at', action='store_true', help='Using a single adversarial perturbation for the whole batch of inputs')
    parser.add_argument('--at_pred_label', action='store_true', help='Use predicted labels for AT.')
    parser.add_argument('--eps', default=0.0, type=float, help='eps for different adversarial training methods')
    parser.add_argument('--sam_rho', default=0.0, type=float, help='perturbation radius for SAM')
    parser.add_argument('--sam_step_size_mult', default=1.0, type=float, help='step size multiplication factor for SAM')
    parser.add_argument('--sam_iters', default=1, type=int, help='number of iterations for SAM')
    parser.add_argument('--sam_lr_mult', default=0.0, type=float, help='0 means no SAM, all other values mean a multiplicative factor to the learning rate.')
    parser.add_argument('--sam_epoch_frac_start', default=0.0, type=float, help='Fraction of epochs when to start SAM (by default 0, i.e. enabling SAM from the very first epoch).')
    parser.add_argument('--sam_epoch_frac_end', default=1.0, type=float, help='Fraction of epochs when to end SAM (by default 1, i.e. enabling SAM until the very last epoch).')
    parser.add_argument('--sam_no_grad_norm', action='store_true', help='no L2 gradient normalization in SAM')
    parser.add_argument('--sam_apply_to_pairing', action='store_true', help='apply SAM to the pairing term (in the first step of SAM)')
    parser.add_argument('--sam_use_prev_batch', action='store_true', help='compute the SAM perturbation on the previous batch')
    parser.add_argument('--only_sam_step_size', action='store_true', help='we modify the step size (grad(w_sam) / grad(w) multiplicative factor) but keep the original gradient direction as in the point `w`.')
    parser.add_argument('--swa_tau', default=0.999, type=float, help='SWA moving averaging coefficient (averaging executed every iteration).')
    parser.add_argument('--add_weight_perturb_scale', default=0.0, type=float, help='std of additive Gaussian perturbations to the weights (mean=0)')
    parser.add_argument('--mul_weight_perturb_scale', default=0.0, type=float, help='std of multiplicative Gaussian perturbations to the weights (mean=1)')
    parser.add_argument('--weight_perturb_distr', default='gauss', choices=['gauss', 'uniform'])
    parser.add_argument('--sgd_p_label_noise', default=0.0, type=float, help='ratio of label noise in SGD per batch')
    parser.add_argument('--attack_init', default='zero', choices=['zero', 'random'])
    parser.add_argument('--attack_iters', default=10, type=int, help='n_iter of pgd for evaluation. warning: the default one is not sufficient for accurate robustness evaluation!')
    parser.add_argument('--pgd_train_n_iters', default=5, type=int, help='n_iter of pgd for training (if attack=pgd)')
    parser.add_argument('--pgd_alpha_train', default=-1, type=float)
    parser.add_argument('--frac_train', default=1, type=float, help='Fraction of training points.')
    parser.add_argument('--acc_steps', default=1, type=int, help='Number of gradient accumulation steps. Ideally args.batch_size should divide this number, otherwise the effective batch size will be smaller.')
    parser.add_argument('--frac_rm_per_batch', default=0, type=float, help='Fraction of training points with the highest loss to be removed from each batch.')
    parser.add_argument('--n_rm_pts', default=0, type=float, help='Fraction of points to remove from the training set.')
    parser.add_argument('--l1_reg', default=0.0, type=float, help='l1 regularization in the objective')
    parser.add_argument('--l2_reg', default=0.0, type=float, help='l2 regularization in the objective')
    parser.add_argument('--pairing_warmup_frac_epochs', default=0.51, type=float, help='Fraction of epochs for a linear warmup for the regularization term.')
    parser.add_argument('--trades_lambda', default=0.0, type=float, help='lambda for TRADES')
    parser.add_argument('--input_grad_reg_lambda', default=0.0, type=float, help='input grad reg lambda')
    parser.add_argument('--weight_grad_reg_lambda', default=0.0, type=float, help='weight grad reg lambda')
    parser.add_argument('--jac_reg_lambda', default=0.0, type=float, help='Jacobian reg lambda')
    parser.add_argument('--l2_pairing_lambda', default=0.0, type=float, help='lambda for the LUO regularization term')
    parser.add_argument('--pairing_noise_std', default=0.0, type=float, help='std of the Gauss noise for the LUO regularization term (warning: then x_paired is not used)')
    parser.add_argument('--kl_pairing_lambda', default=0.0, type=float, help='lambda for the KL regularization term (inspired by TRADES but without an adversarial optimization part)')
    parser.add_argument('--clean_loss_coeff', default=0.0, type=float, help='multiplier for the clean loss term (calculated using the same batch size as the main part)')
    parser.add_argument('--loss_offset', default=0.0, type=float, help='loss offset')
    parser.add_argument('--gce_q', default=-1.0, type=float, help='parameter q of the generalized cross-entropy loss')
    parser.add_argument('--grad_gauss_std', default=0.0, type=float, help='std of Gaussian noise added to each gradient')
    parser.add_argument('--scale_init', default=0.0, type=float, help='scale init for *linear* models')
    parser.add_argument('--droprate', default=0.0, type=float, help='dropout rate')
    parser.add_argument('--epoch_rm_noise', default=-1, type=float, help='remove noise at epoch epoch_rm_noise*epochs')
    parser.add_argument('--activation', default='relu', type=str, help='currently supported only for resnet. relu or softplusA where A corresponds to the softplus alpha')
    parser.add_argument('--rm_criterion', default='loss', choices=['loss', 'entropy'], type=str, help='Points with the highest values of this metric will be removed from the train set.')
    parser.add_argument('--loss', default='ce', choices=['ce', 'ce_offset', 'gce', 'smallest_k_ce'], type=str, help='Loss type.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--alpha_weights_linear_at', action='store_true')
    parser.add_argument('--alpha_weights_linear_at1', action='store_true')
    parser.add_argument('--alpha_weights_linear_at2', action='store_true')
    parser.add_argument('--normalize_x', action='store_true', help='l2 normalization of input images')
    parser.add_argument('--export_images', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--final_adv_eval', action='store_true')
    parser.add_argument('--save_model_each_k_epochs', default=0, type=int, help='save each k epochs; 0 means saving only at the end')
    parser.add_argument('--half_prec', action='store_true', help='if enabled, runs everything as half precision [not recommended]')
    parser.add_argument('--no_data_augm', action='store_true')
    parser.add_argument('--freeze_bn_stats', action='store_true')
    parser.add_argument('--eval_early_stopped_model', action='store_true', help='whether to evaluate the model obtained via early stopping')
    parser.add_argument('--eval_iter_freq', default=-1, type=int, help='how often to evaluate test stats. -1 means to evaluate each #iter that corresponds to 2nd epoch with frac_train=1.')
    parser.add_argument('--n_eval_every_k_iter', default=512, type=int, help='on how many examples to eval every k iters')
    parser.add_argument('--n_layers', default=1, type=int, help='#layers (for model in [fc, cnn])')
    parser.add_argument('--model_width', default=-1, type=int, help='model width (# conv filters on the first layer for ResNets)')
    parser.add_argument('--batch_size_eval', default=512, type=int, help='batch size for the final eval with pgd rr; 6 GB memory is consumed for 1024 examples with fp32 network')
    parser.add_argument('--n_final_eval', default=10000, type=int, help='on how many examples to do the final evaluation; -1 means on all test examples.')
    parser.add_argument('--exp_name', default='other', type=str)
    parser.add_argument('--rlat_layers', default='lpips', choices=['all', 'lpips', 'bnonly', 'convonly', 'block0', 'block1', 'block2', 'block3', 'block4'], type=str)
    parser.add_argument('--rlat_dim_scaling', default='multiply', choices=['multiply', 'divide', 'divide_sqrt', 'const'], type=str)
    parser.add_argument('--rlat_layer_scaling', default='descending', choices=['const', 'descending', 'ascending', 'lindescending'], type=str)
    parser.add_argument('--model_path', type=str, default='', help='Path to a checkpoint to continue training from.')
    return parser.parse_args()


def main():
    args = get_args()
    assert args.model_width != -1, 'args.model_width has to be always specified (e.g., 64 for resnet18, 10 for wrn28)'
    if args.activation == 'softplus':  # only implemented for resnet18 currently
        assert args.model == 'resnet18'
    if args.attack == 'pgd':
        assert args.pgd_alpha_train != -1, 'set --pgd_alpha_train for custom pgd AT'
    assert 0 <= args.frac_train <= 1
    assert 0 <= args.sgd_p_label_noise <= 1
    if args.trades_lambda > 0:
        assert args.eps > 0
    if args.pairing_noise_std > 0:
        assert args.l2_pairing_lambda > 0 or args.kl_pairing_lambda > 0
    assert not (args.sam_rho != 0 and args.sam_lr_mult != 0)  # only one of them should be specified
    assert 0 <= args.swa_tau <= 1

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print('GPU memory available: {:.2f} GB'.format(torch.cuda.get_device_properties('cuda').total_memory / 10**9))
    cur_timestamp = str(datetime.now())[:-3]  # include also ms to prevent the probability of name collision
    model_name = '{} dataset={} model={} epochs={} lr_max={} model_width={} l2_reg={} sam_rho={} batch_size={} acc_steps={} frac_train={} p_label_noise={} seed={}'.format(
        cur_timestamp, args.dataset, args.model, args.epochs, args.lr_max, args.model_width, args.l2_reg, 
        args.sam_rho, args.batch_size, args.acc_steps, args.frac_train, args.p_label_noise, args.seed)
    logger = utils.configure_logger(model_name, args.debug)
    logger.info(args)

    eps = args.eps
    n_cls = 2 if args.dataset in ['cifar10_horse_car', 'cifar10_dog_cat'] else 10 if args.dataset != 'cifar100' else 100
    n_train = int(args.frac_train * data.shapes_dict[args.dataset][0])
    n_train_effective = n_train if n_train != -1 else data.shapes_dict[args.dataset][0]
    frac_val_set = 0.1
    n_val = int(frac_val_set * n_train_effective)
    n_train_effective = int((1-frac_val_set) * n_train_effective)

    args.exp_name = 'exps/{}'.format(args.exp_name)
    if not os.path.exists(args.exp_name): os.makedirs(args.exp_name)

    # fixing the seed helps, but not completely. there is still some non-determinism due to GPU computations.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    pgd_alpha = eps / 4  # may seem too large e.g. for eps=8, but see the ablation study in Croce and Hein, 2020
    if args.at_norm == 'linf':  # divide by 255 only for linf; note that --at_norm=linf by default
        eps, pgd_alpha, args.pgd_alpha_train = args.eps / 255, pgd_alpha / 255, args.pgd_alpha_train / 255
    else:
        eps, pgd_alpha, args.pgd_alpha_train = args.eps, pgd_alpha, args.pgd_alpha_train

    if args.attack in ['fgsm', 'fgsmpp']:
        n_steps_pgd_curr, step_size_pgd_curr = 1, eps  # we start from these values and then adapt them
    else:
        n_steps_pgd_curr, step_size_pgd_curr = args.pgd_train_n_iters, args.pgd_alpha_train  # otherwise these values are fixed

    val_indices = np.random.permutation(data.shapes_dict[args.dataset][0])[:n_val]
    train_data_augm = False if args.no_data_augm or args.model == 'linear' or args.dataset in ['mnist', 'mnist_binary', 'gaussians_binary'] else True
    train_batches = data.get_loaders(args.dataset, n_train, args.batch_size, split='train', val_indices=val_indices, shuffle=True, data_augm=train_data_augm, p_label_noise=args.p_label_noise, noise_type=args.noise_type, drop_last=True)
    train_batches_fast = data.get_loaders(args.dataset, args.n_eval_every_k_iter, args.batch_size, split='train', val_indices=val_indices, shuffle=False, data_augm=False, p_label_noise=args.p_label_noise, noise_type=args.noise_type, drop_last=False)
    val_batches = data.get_loaders(args.dataset, n_val, args.batch_size, split='val', val_indices=val_indices, shuffle=True, data_augm=False, p_label_noise=args.p_label_noise, noise_type=args.noise_type, drop_last=False)
    test_batches = data.get_loaders(args.dataset, args.n_final_eval, args.batch_size_eval, split='test', shuffle=True, data_augm=False, noise_type=args.noise_type, drop_last=False)
    test_batches_fast = data.get_loaders(args.dataset, args.n_eval_every_k_iter, args.batch_size_eval, split='test', shuffle=True, data_augm=False, noise_type=args.noise_type, drop_last=False)

    model = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width,
                             args.activation, droprate=args.droprate).cuda()
    # print(sum([np.prod(param.shape) for param in model.parameters()]))  # print the total number of parameters
    if args.model_path != '':
        model_dict = torch.load(args.model_path)['last']
        model.load_state_dict({k: v for k, v in model_dict.items() if 'model_preact_hl1' not in k})
    else:
        model.apply(models.init_weights(args.model, args.scale_init))
    model.train()
    model_swa = copy.deepcopy(model).eval()  # stochastic weight averaging model (keep it in the eval mode by default)

    if args.grad_gauss_std > 0:
        for param in model.parameters():  # add noise directly to the gradient
            param.register_hook(lambda grad: grad + args.grad_gauss_std * torch.randn(grad.shape).cuda())

    if args.only_sam_step_size:
        raise NotImplementedError('new perturbation function does not support this yet')
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=model.half_prec)
    lr_schedule = utils_train.get_lr_schedule(args.lr_schedule, args.epochs, args.lr_max)
    ln_schedule = utils_train.get_lr_schedule(args.ln_schedule, args.epochs, args.sgd_p_label_noise)

    loss_dict = {
        'ce': losses.cross_entropy(),
        'ce_offset': losses.cross_entropy_with_offset(args.loss_offset),
        'gce': losses.generalized_cross_entropy(args.gce_q),
        'smallest_k_ce': losses.smallest_k_cross_entropy(args.frac_rm_per_batch)
    }
    loss_f = loss_dict[args.loss]
    if args.loss == 'ce' and n_cls == 2:
        loss_f = losses.logistic_loss()
    # normalize by the batch size but not by the number of classes
    loss_f_trades = torch.nn.KLDivLoss(reduction='batchmean')
    if args.loss != 'ce_offset' and args.loss_offset > 0.0:
        raise ValueError('Perhaps --loss=ce_offset should be used.')
    if args.loss != 'gce' and args.gce_q != -1.0:
        raise ValueError('Perhaps --loss=gce should be used.')

    metr_dict = defaultdict(list, vars(args))
    val_err_best, best_state_dict = np.inf, copy.deepcopy(model.state_dict())
    val_swa_err_best, best_swa_state_dict = np.inf, copy.deepcopy(model.state_dict())
    start_time = time.time()
    time_train, iteration, last_epoch_cat_overfit = 0, 0, 0
    train_obj, train_reg, train_n_ln, train_n_clean, train_err_ln = 0, 0, 0, 0, 0
    margin_clean, margin_ln, l2_norm_w = 0, 0, 0
    x_prev, y_prev = torch.zeros([args.batch_size, 3, 32, 32]).cuda(), torch.zeros(args.batch_size, dtype=torch.int64).cuda()  # only for the very first iteration
    for epoch in range(args.epochs + 1):
        if epoch == int(args.epoch_rm_noise * args.epochs):
            train_batches = data.get_loaders(args.dataset, -1, args.batch_size, split='train', shuffle=False,
                                             data_augm=False, p_label_noise=args.p_label_noise, drop_last=False)
            _, logits = utils_eval.get_logits(train_batches, model, eps, pgd_alpha, scaler, args.attack_iters, adversarial=False)
            x_train_adv, logits_adv = utils_eval.get_logits(train_batches, model, eps, pgd_alpha, scaler, args.attack_iters, adversarial=True)
            x_train, y_train, y_train_correct, ln_train = data.get_xy_from_loader(train_batches, cuda=False)
            loss_vals = F.cross_entropy(logits_adv, y_train.cpu(), reduction='none')
            probs = F.softmax(logits_adv)
            entropies = -(probs * torch.log(probs)).sum(1)
            if args.export_images:
                np.save('{}/images_{}.npy'.format(args.exp_name, model_name),
                        {'x_train': x_train, 'y_train': y_train, 'x_train_adv': x_train_adv, 'logits': logits, 'losses': loss_vals, 'entropies': entropies})

            n_pts_keep = int((1 - args.n_rm_pts) * len(loss_vals))
            idx_all = np.argsort(loss_vals if args.rm_criterion == 'loss' else entropies)
            idx_keep, idx_rm = idx_all[:n_pts_keep], idx_all[n_pts_keep:]
            idx_keep = np.random.permutation(idx_keep)  # needed to make the first k examples i.i.d. for train_batches_fast (but now there is no train_batches_fast)

            x_train, y_train, ln_train = x_train[idx_keep].cpu(), y_train[idx_keep].cpu(), ln_train[idx_keep].cpu()
            train_batches = data.create_loader(x_train, y_train, ln_train, -1, args.batch_size, shuffle=True, drop_last=True)
            print('Removed {} pts, avg loss: {:.2f} kept vs {:.2f} removed'.format(
                  len(idx_rm), loss_vals[idx_keep].mean(), loss_vals[idx_rm].mean()))

        model = model.eval() if epoch == 0 else model.train()  # epoch=0 is eval only

        for i, (x, x_paired, y, _, ln) in enumerate(train_batches):
            if epoch == 0 and i > 0:  # epoch=0 runs only for one iteration (to check the training stats at init)
                break
            time_start_iter = time.time()
            n_clean, n_ln = y[~ln].size(0), y[ln].size(0)
            x, x_paired, y = x.cuda(), x_paired.cuda(), y.cuda()
            lr = lr_schedule(epoch - 1 + (i + 1) / len(train_batches))  # epoch - 1 since the 0th epoch is skipped
            opt.param_groups[0].update(lr=lr)

            if args.sgd_p_label_noise > 0.0:
                sgd_p_label_noise_eff = ln_schedule(epoch - 1 + (i + 1) / len(train_batches))
                # n_noisy_pts = int(args.batch_size * sgd_p_label_noise_eff)  # fixed fraction of noisy points
                n_noisy_pts = (torch.rand(args.batch_size) < sgd_p_label_noise_eff).int().sum()  # randomized fraction of noisy points
                rand_indices = torch.randperm(args.batch_size)[:n_noisy_pts].cuda()
                rand_labels = torch.randint(low=0, high=n_cls, size=(n_noisy_pts, )).cuda()
                y[rand_indices] = rand_labels

            if args.add_weight_perturb_scale > 0 or args.mul_weight_perturb_scale > 0:
                weights_delta_dict = utils_train.perturb_weights(
                    model, args.add_weight_perturb_scale, args.mul_weight_perturb_scale, args.weight_perturb_distr)

            def fp_loss_reg(x, y, delta, bn_stats=True, per_example_weights=None):
                utils_train.change_bn_mode(model, bn_stats)

                if args.input_grad_reg_lambda > 0 or args.jac_reg_lambda > 0:  # to enable double backprop
                    delta.requires_grad = True

                with torch.cuda.amp.autocast(enabled=model.half_prec):
                    logits = model(x + delta) 
                    if per_example_weights is None:
                        obj = loss_f(logits, y)
                    else:
                        obj = torch.mean(per_example_weights * loss_f(logits, y, reduction=False))

                if args.clean_loss_coeff > 0.0:  # 50% clean and 50% adv, but the coefficient may vary
                    assert args.attack != 'none'  # none => same as normal training with a larger batch size (not supposed usage)
                    with torch.cuda.amp.autocast(enabled=model.half_prec):
                        obj += args.clean_loss_coeff * loss_f(model(x), y) 

                reg = torch.zeros(1).cuda()[0]
                model_params = list(model.named_parameters())
                # if not args.sam_w_sam_prime and not args.sam_w_prime:
                for i_param, (param_name, param) in enumerate(model_params):
                    # if 'bn' in param_name or 'bias' in param_name or len(param.shape) == 1:  # skip BN and biases from WD
                    #     continue
                    reg += args.l1_reg * torch.sum(torch.abs(param)).float()
                    reg += args.l2_reg * 0.5 * torch.sum(param ** 2).float()  # we multiply by 0.5 as in the SAM paper.

                if args.jac_reg_lambda > 0:
                    for i in range(n_cls):
                        grad_i = torch.autograd.grad(scaler.scale(logits[:, i].mean()), delta, create_graph=True)[0]
                        grad_i = grad_i / scaler.get_scale()
                        with torch.cuda.amp.autocast(enabled=model.half_prec):
                            reg += args.jac_reg_lambda * torch.sum(grad_i ** 2)

                utils_train.change_bn_mode(model, True)

                obj += reg
                return obj / args.acc_steps, reg / args.acc_steps, delta, logits

            accum_grad_dict = defaultdict(lambda: 0)  # dict for gradient accumulation
            acc_batch_size = args.batch_size // args.acc_steps
            for acc_step in range(args.acc_steps):
                batch_start, batch_end = acc_step * acc_batch_size, (acc_step + 1) * acc_batch_size
                x_curr, y_curr = x[batch_start:batch_end], y[batch_start:batch_end]
                delta_curr = torch.zeros_like(x_curr, requires_grad=False)  # initialize delta_curr
                if (args.sam_rho != 0 or args.sam_lr_mult != 0) and args.sam_epoch_frac_start <= epoch/args.epochs <= args.sam_epoch_frac_end:
                    if args.sam_lr_mult != 0:
                        rho_actual = lr * args.sam_lr_mult
                    else:
                        rho_actual = args.sam_rho

                    if args.sam_use_prev_batch:  # i.e., MaxSum SAM
                        x_curr_sam, y_curr_sam = x_prev[batch_start:batch_end], y_prev[batch_start:batch_end]
                    else:  # closer to SumMax SAM
                        x_curr_sam, y_curr_sam = x_curr, y_curr
                    fp_loss_sam = lambda model: fp_loss_reg(x_curr_sam, y_curr_sam, delta_curr, bn_stats=not args.freeze_bn_stats)[0]
                    sam_delta_dict = perturb_weights_sam(model, fp_loss_sam, rho_actual, rho_actual*args.sam_step_size_mult, args.sam_iters, args.sam_no_grad_norm)
                    
                    obj, reg, delta_curr, logits = fp_loss_reg(x_curr, y_curr, delta_curr, bn_stats=not args.freeze_bn_stats)
                    scaler.scale(obj).backward()
                    utils_train.subtract_weight_delta(model, sam_delta_dict)  # return back to `w_t` after computing the gradient at `w_t + delta_t`

                else:
                    obj, reg, delta_curr, _ = fp_loss_reg(x_curr, y_curr, delta_curr, bn_stats=not args.freeze_bn_stats)
                    scaler.scale(obj).backward(create_graph=True if args.weight_grad_reg_lambda > 0 else False)

                    if args.add_weight_perturb_scale > 0 or args.mul_weight_perturb_scale > 0:
                        utils_train.subtract_weight_delta(model, weights_delta_dict)

                    if args.weight_grad_reg_lambda > 0:
                        reg_weight_grad_reg = torch.zeros(1).cuda()[0]
                        for param in model.parameters():
                            if param.grad is not None:
                                reg_weight_grad_reg += args.weight_grad_reg_lambda * torch.sum(param.grad ** 2)
                        obj, reg = obj + reg_weight_grad_reg, reg + reg_weight_grad_reg  # needed only for statistics
                        obj += reg_weight_grad_reg  # this is needed only for printing the statistics
                        # add the grads of the weight grad loss term only (previous grads are already in param.grads)
                        scaler.scale(reg_weight_grad_reg).backward()

                for param in model.parameters():
                    if param.grad is not None:
                        accum_grad_dict[param] += param.grad
                opt.zero_grad()

                train_obj += obj.item() * y.size(0)  # only for statistics
                train_reg += reg.item() * y.size(0)  # only for statistics

            if epoch > 0:  # on 0-th epoch only evaluation occurs
                # put the final grads from accum_grad_dict to param.grads
                for param in accum_grad_dict.keys():
                    param.grad = accum_grad_dict[param]
                if args.model == 'linear':
                    utils_train.modify_grads_lin_model(model, x, y, eps, args)

                # grad_norm = sum([torch.sum(p.grad**2) for p in model.parameters()])**0.5
                # print(grad_norm)
                scaler.step(opt)
                scaler.update()  # update the scale of the loss for fp16

            opt.zero_grad()  # zero grad (also at epoch==0)

            time_train += time.time() - time_start_iter
            train_n_clean += n_clean
            train_n_ln += n_ln

            if args.model == 'linear':
                y_plus_minus = 2 * (y - 0.5)
                w = model._model[1].weight
                w_l2_norm = (w ** 2).sum() ** 0.5
                margin_clean += (y_plus_minus[~ln, None]*logits[~ln] / w_l2_norm).mean().item()
                margin_ln += (y_plus_minus[ln, None]*logits[ln] / w_l2_norm).mean().item()
                l2_norm_w = ((w**2).sum()**0.5).item()

            utils_train.moving_average(model_swa, model, 1-args.swa_tau)  # executed every iteration

            # by default, evaluate every 2 epochs (update: 5 temporary to save time)
            if (args.eval_iter_freq == -1 and iteration % (5 * (n_train_effective // args.batch_size)) == 0) or \
               (args.eval_iter_freq != -1 and iteration % args.eval_iter_freq == 0):
                utils_train.bn_update(train_batches, model_swa)  # a bit heavy but ok to do once per 2 epochs

                model.eval()  # it'd be incorrect to recalculate the BN stats based on some evaluations

                train_obj, train_reg = train_obj / (train_n_ln+train_n_clean), train_reg / (train_n_ln+train_n_clean)

                train_err_clean, _, _ = utils_eval.rob_err(train_batches, model, eps, pgd_alpha, scaler, 0, 0, noisy_examples='none', loss_f=loss_f, n_batches=4)  # i.e. it's evaluated using 4*batch_size examples
                train_err, train_loss, _ = utils_eval.rob_err(train_batches, model, eps, pgd_alpha, scaler, 0, 0, loss_f=loss_f, n_batches=4)  # i.e. it's evaluated using 4*batch_size examples
                train_err_swa, train_loss_swa, _ = utils_eval.rob_err(train_batches, model_swa, eps, pgd_alpha, scaler, 0, 0, loss_f=loss_f, n_batches=4)  # i.e. it's evaluated using 4*batch_size examples

                if args.p_label_noise > 0.0:
                    train_err_ln, _, _ = utils_eval.rob_err(train_batches, model, eps, pgd_alpha, scaler, 0, 0, noisy_examples='all', loss_f=loss_f, n_batches=4)

                norm_grad_clean, norm_grad_ln, norm_grad_all, norm_grad_all_sam = 0, 0, 0, 0
                cos_grads_clean_all, cos_grads_all_all_sam, cos_grads_clean_all_sam, cos_grads_ln_all, cos_grads_ln_all_sam = 0, 0, 0, 0, 0

                sharpness_rand_all = 0  # utils_eval.eval_avg_sharpness(model, scaler, train_batches, noisy_examples='default', sigma=0.1 if 'resnet' in args.model else 0.02)
                # Note 1: for computing sharpness, the large rho means the same as during training (except if it's ERM, then it's fixed to 0.1)
                # Note 2: `train_batches_fast` contains samples *without* augmentations to prevent any stochasticity over different sharpness computations
                sam_rho_small, sam_rho_large = 0.01, args.sam_rho if args.sam_rho > 0.0 else 0.1
                train_err, train_loss, _ = utils_eval.rob_err(train_batches_fast, model, eps, pgd_alpha, scaler, 0, 0, loss_f=loss_f) 
                sharpness_loss_small_fast, sharpness_err_small_fast, _ = utils_eval.eval_sharpness(model, train_batches_fast, loss_f, sam_rho_small, 1.0, 1, 1, 'all', rand_init=False)
                sharpness_loss_small_slow, sharpness_err_small_slow = 0.0, 0.0
                # sharpness_loss_small_slow, sharpness_err_small_slow, _ = utils_eval.eval_sharpness(model, train_batches_fast, loss_f, sam_rho_small, 0.1, 100, 1, 'all', rand_init=False)
                sharpness_loss_large_fast, sharpness_err_large_fast, _ = utils_eval.eval_sharpness(model, train_batches_fast, loss_f, sam_rho_large, 1.0, 1, 1, 'all', rand_init=False)
                # sharpness_loss_large_slow, sharpness_err_large_slow, _ = utils_eval.eval_sharpness(model, train_batches_fast, loss_f, sam_rho_large, 0.1, 100, 1, 'all', rand_init=False)
                sharpness_loss_large_slow, sharpness_err_large_slow = 0.0, 0.0
                max_loss_small_fast, max_loss_small_slow = sharpness_loss_small_fast + train_loss, sharpness_loss_small_slow + train_loss
                max_err_small_fast, max_err_small_slow = sharpness_err_small_fast + train_err, sharpness_err_small_slow + train_err
                max_loss_large_fast, max_loss_large_slow = sharpness_loss_large_fast + train_loss, sharpness_loss_large_slow + train_loss
                max_err_large_fast, max_err_large_slow = sharpness_err_large_fast + train_err, sharpness_err_large_slow + train_err 

                val_err, _, _ = utils_eval.rob_err(val_batches, model, eps, pgd_alpha, scaler, 0, 0, loss_f=loss_f)
                val_err_swa, _, _ = utils_eval.rob_err(val_batches, model_swa, eps, pgd_alpha, scaler, 0, 0, loss_f=loss_f)
                if args.attack != 'none':  # we need to compute both to detect catastrophic overfitting or do model selection
                    train_err_pgd, train_loss_pgd, _ = utils_eval.rob_err(train_batches, model, eps, pgd_alpha, scaler, args.attack_iters, 1, n_batches=4)
                    val_err_pgd, _, _ = utils_eval.rob_err(val_batches, model, eps, pgd_alpha, scaler, args.attack_iters, 1, n_batches=2)
                    val_err_fgsm, _, _ = utils_eval.rob_err(val_batches, model, eps, eps, scaler, 1, 1, n_batches=2)
                    val_err_swa_pgd, _, _ = utils_eval.rob_err(val_batches, model_swa, eps, pgd_alpha, scaler, args.attack_iters, 1, n_batches=2)
                    test_err_pgd, test_loss_pgd, _ = utils_eval.rob_err(test_batches_fast, model, eps, pgd_alpha, scaler, args.attack_iters, 1)
                else:
                    train_err_pgd, train_loss_pgd, val_err_pgd, val_err_fgsm, test_err_pgd, test_loss_pgd = 0, 0, 0, 0, 0, 0

                test_err, test_loss, _ = utils_eval.rob_err(test_batches, model, eps, pgd_alpha, scaler, 0, 0, loss_f=loss_f)
                test_err_swa, _, _ = utils_eval.rob_err(test_batches, model_swa, eps, pgd_alpha, scaler, 0, 0, loss_f=loss_f)
                test_err_c10c = 0.0

                time_elapsed = time.time() - start_time
                linear_str = '|w|_2 {:.2f} margin_clean {:.2f}'.format(l2_norm_w, margin_clean) if args.model == 'linear' else ''
                train_str = '[train] obj {:.3f} reg {:.3f} loss {:.4f}/{:.4f} err_clean {:.2%} err_ln {:.2%} {}'.format(
                    train_obj, train_reg, train_loss, train_loss_swa, train_err_clean, train_err_ln, linear_str)
                test_str = '[test] err {:.2%}/{:.2%} '.format(test_err, test_err_swa)
                val_str = '[val] err {:.2%}/{:.2%}'.format(val_err, val_err_swa)
                rob_str = '[rob] err {:.2%}/{:.2%}/{:.2%}'.format(train_err_pgd, val_err_pgd, test_err_pgd) if args.attack != 'none' else ''
                # cos_str = 'cos {:.3f}-{:.3f}'.format(cos_grads_clean_all, cos_grads_clean_all_sam)
                # norm_str = 'norm {:.3f}-{:.3f}-{:.3f}'.format(norm_grad_clean, norm_grad_all, norm_grad_all_sam)
                sharpness_str = 'sharp {:.3f}-{:.3f} {:.3f}-{:.3f}'.format(sharpness_loss_small_fast, sharpness_loss_small_slow, sharpness_loss_large_fast, sharpness_loss_large_slow)
                logger.info('{}-{}: {}  {}  {}  {}  {} ({:.2f}m, {:.2f}m)'.format(
                    epoch, iteration, train_str, test_str, val_str, rob_str, sharpness_str, time_train/60, time_elapsed/60))
                metr_vals = [epoch, iteration, train_obj, train_loss, train_reg, train_err_clean, train_err_ln,
                             train_err_pgd, test_err, test_loss, train_loss_swa, train_err_swa, val_err_swa,
                             test_err_swa, test_err_pgd, test_loss_pgd, train_loss_pgd, l2_norm_w,
                             margin_clean, margin_ln, sharpness_rand_all, 
                             sharpness_loss_small_fast, sharpness_err_small_fast, sharpness_loss_small_slow, sharpness_err_small_slow,
                             sharpness_loss_large_fast, sharpness_err_large_fast, sharpness_loss_large_slow, sharpness_err_large_slow, 
                             max_loss_small_fast, max_err_small_fast, max_loss_small_slow, max_err_small_slow,
                             max_loss_large_fast, max_err_large_fast, max_loss_large_slow, max_err_large_slow,
                             test_err_c10c, val_err, val_err_pgd, time_train, time_elapsed,
                             norm_grad_clean, norm_grad_ln, norm_grad_all, norm_grad_all_sam,
                             cos_grads_clean_all, cos_grads_all_all_sam, cos_grads_clean_all_sam,
                             cos_grads_ln_all, cos_grads_ln_all_sam]
                metr_names = ['epoch', 'iter', 'train_obj', 'train_loss', 'train_reg', 'train_err_clean',
                              'train_err_ln', 'train_err_pgd', 'test_err', 'test_loss',
                              'train_loss_swa', 'train_err_swa', 'val_err_swa', 'test_err_swa',
                              'test_err_pgd', 'test_loss_pgd', 'train_loss_pgd',
                              'l2_norm_w', 'margin_clean', 'margin_ln', 'sharpness_rand_all', 
                              'sharpness_loss_small_fast', 'sharpness_err_small_fast', 'sharpness_loss_small_slow', 'sharpness_err_small_slow',
                              'sharpness_loss_large_fast', 'sharpness_err_large_fast', 'sharpness_loss_large_slow', 'sharpness_err_large_slow', 
                              'max_loss_small_fast', 'max_err_small_fast', 'max_loss_small_slow', 'max_err_small_slow',
                              'max_loss_large_fast', 'max_err_large_fast', 'max_loss_large_slow', 'max_err_large_slow',
                              'test_err_c10c', 'val_err', 'val_err_pgd', 'time_train', 'time_elapsed',
                              'norm_grad_clean', 'norm_grad_ln', 'norm_grad_all', 'norm_grad_all_sam',
                              'cos_grads_clean_all', 'cos_grads_all_all_sam', 'cos_grads_clean_all_sam',
                              'cos_grads_ln_all', 'cos_grads_ln_all_sam']
                utils.update_metrics(metr_dict, metr_vals, metr_names)

                if not args.debug:
                    np.save('{}/{}.npy'.format(args.exp_name, model_name), metr_dict)

                if args.attack == 'none':
                    if val_err < val_err_best:  # save the best model according to val_err
                        val_err_best, best_state_dict = val_err, copy.deepcopy(model.state_dict())
                    if val_err_swa < val_swa_err_best:  # save the best model according to val_err_swa
                        val_swa_err_best, best_swa_state_dict = val_err_swa, copy.deepcopy(model_swa.state_dict())
                else:
                    if val_err_pgd < val_err_best:  # save the best model according to val_err
                        val_err_best, best_state_dict = val_err_pgd, copy.deepcopy(model.state_dict())
                    if val_err_swa_pgd < val_swa_err_best:  # save the best model according to val_err_swa
                        val_swa_err_best, best_swa_state_dict = val_err_swa_pgd, copy.deepcopy(model_swa.state_dict())

                # increase #steps and decrease the step size if
                # (1) the gap between the PGD and FGSM error is >= 30%
                # (2) don't do this more often than once per 5 epochs
                # (3) don't increase beyond 5 steps
                if args.attack == 'fgsmpp' and val_err_pgd - val_err_fgsm >= 0.3 and epoch - last_epoch_cat_overfit >= 5 and n_steps_pgd_curr <= 4:
                    last_epoch_cat_overfit = epoch
                    n_steps_pgd_curr += 1
                    step_size_pgd_curr = eps / n_steps_pgd_curr
                    logger.info('update n_steps_pgd_curr={}, step_size_pgd_curr={}'.format(n_steps_pgd_curr, step_size_pgd_curr))

                model.train()
                train_obj, train_reg, train_err_clean, train_err_ln, train_n_clean, train_n_ln = 0, 0, 0, 0, 0, 0
                margin_clean, margin_ln = 0, 0

            x_prev, y_prev = x, y
            iteration += 1

        if args.save_model_each_k_epochs > 0:
            if epoch % args.save_model_each_k_epochs == 0 or epoch <= 5:
                torch.save({'last': model.state_dict()}, 'models/{} epoch={}.pth'.format(model_name, epoch))

        if not args.debug:
            np.save('{}/{}.npy'.format(args.exp_name, model_name), metr_dict)
            if epoch == args.epochs:  # only save at the end
                torch.save({'last': model.state_dict(), 'best': best_state_dict,
                            'swa_last': model_swa.state_dict(), 'swa_best': best_swa_state_dict},
                            'models/{} epoch={}.pth'.format(model_name, epoch))

    best_eval_val = np.argmin(metr_dict['val_err'])
    best_eval_test = np.argmin(metr_dict['test_err'])
    logger.info('Min values: test_err {:.2%} ({}-th epoch) [test set model selection]'.format(
        metr_dict['test_err'][best_eval_test], metr_dict['epoch'][best_eval_test]))
    logger.info('Min values: test_err {:.2%} ({}-th epoch) [val set model selection]'.format(
        metr_dict['test_err'][best_eval_val], metr_dict['epoch'][best_eval_val]))

    best_eval_val_swa = np.argmin(metr_dict['val_err_swa'])
    best_eval_test_swa = np.argmin(metr_dict['test_err_swa'])
    logger.info('Min values SWA: test_err {:.2%} ({}-th epoch) [test set model selection]'.format(
        metr_dict['test_err_swa'][best_eval_test_swa], metr_dict['epoch'][best_eval_test_swa]))
    logger.info('Min values SWA: test_err {:.2%} ({}-th epoch) [val set model selection]'.format(
        metr_dict['test_err_swa'][best_eval_val_swa], metr_dict['epoch'][best_eval_val_swa]))

    best_eval_pgd_val = np.argmin(metr_dict['val_err_pgd'])
    best_eval_pgd_test = np.argmin(metr_dict['test_err_pgd'])
    logger.info('Min values: pgd_test_err {:.2%} ({}-th epoch) [test set model selection]'.format(
        metr_dict['test_err_pgd'][best_eval_pgd_test], metr_dict['epoch'][best_eval_pgd_test]))
    logger.info('Min values: pgd_test_err {:.2%} ({}-th epoch) [val set model selection]'.format(
        metr_dict['test_err_pgd'][best_eval_pgd_val], metr_dict['epoch'][best_eval_pgd_val]))

    logger.info('Saved the model at: models/{} epoch={}.pth'.format(model_name, epoch))
    logger.info('Done in {:.2f}m'.format((time.time() - start_time) / 60))


if __name__ == "__main__":
    main()
