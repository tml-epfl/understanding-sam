import argparse
import os
import copy
import numpy as np
import torch
import time
import data
import models
import losses
import utils_eval
import utils_train


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar10_horse_car', 'cifar10_dog_cat', 'uniform_noise'], type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'cnn', 'fc', 'linear', 'lenet'], type=str)
    parser.add_argument('--model_name1', type=str, help='model name 1')
    parser.add_argument('--model_name2', type=str, help='model name 2')
    parser.add_argument('--model_path_erm', type=str, help='model name 1')
    parser.add_argument('--model_path_sam', type=str, help='model name 2')
    parser.add_argument('--early_stopped_model_erm', action='store_true', help='eval the best model according to acc evaluated every k iters (typically, k=200)')
    parser.add_argument('--early_stopped_model_sam', action='store_true', help='eval the best model according to acc evaluated every k iters (typically, k=200)')
    parser.add_argument('--label', default='noname', type=str, help='label to append to the exported file')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n_eval', default=1024, type=int, help='#examples to evaluate on')
    parser.add_argument('--n_eval_sharpness', default=256, type=int, help='#examples to evaluate on')
    parser.add_argument('--bs_sharpness', default=128, type=int, help='batch size for sharpness experiments')
    parser.add_argument('--loss', default='ce', choices=['ce', 'ce_offset', 'gce', 'smallest_k_ce'], type=str, help='Loss type.')
    parser.add_argument('--sam_rho', default=0.2, type=float, help='step size for SAM (sharpness-aware minimization)')
    parser.add_argument('--p_label_noise', default=0.0, type=float, help='label noise for evaluation')
    parser.add_argument('--activation', default='relu', type=str, help='currently supported only for resnet. relu or softplus* where * corresponds to the softplus alpha')
    parser.add_argument('--model_width', default=64, type=int, help='model width (# conv filters on the first layer for ResNets)')
    parser.add_argument('--distance_only', action='store_true', help='no evaluation of the loss surfaces')
    parser.add_argument('--half_prec', action='store_true', help='eval in half precision')
    parser.add_argument('--aa_eval', action='store_true', help='perform autoattack evaluation')
    return parser.parse_args()


start_time = time.time()
args = get_args()
print_stats = False
n_cls = 2 if args.dataset in ['cifar10_horse_car', 'cifar10_dog_cat'] else 10
p_label_noise = args.model_path_erm.split('p_label_noise=')[1].split(' ')[0]
assert p_label_noise == args.model_path_sam.split('p_label_noise=')[1].split(' ')[0], 'ln level should be the same for the visualization'
assert args.n_eval_sharpness % args.bs_sharpness == 0, 'args.n_eval should be divisible by args.bs_sharpness'

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

np.set_printoptions(precision=4, suppress=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

scaler = torch.cuda.amp.GradScaler(enabled=False)
loss_dict = {
    'ce': losses.cross_entropy(),
    'ce_offset': losses.cross_entropy_with_offset(loss_offset=0.1),
    'gce': losses.generalized_cross_entropy(q=0.7),
    'smallest_k_ce': losses.smallest_k_cross_entropy(frac_rm_per_batch=0.0)
}
loss_f = loss_dict[args.loss]

model_init1 = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
model_init1.apply(models.init_weights(args.model))
model_init2 = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
model_init2.apply(models.init_weights(args.model))
model_interpolated = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
model_interpolated.apply(models.init_weights(args.model))
model_erm = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
model_erm_swa = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
model_sam = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()
model_sam_swa = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation).cuda().eval()

model_erm_dict_orig = torch.load('models/{}.pth'.format(args.model_path_erm))
model_erm_dict = model_erm_dict_orig['best'] if args.early_stopped_model_erm else model_erm_dict_orig['last']
model_erm.load_state_dict({k: v for k, v in model_erm_dict.items()})
if 'swa_best' in model_erm_dict_orig:
    model_erm_swa_dict = model_erm_dict_orig['swa_best']  # if args.early_stopped_model_erm else model_erm_dict_orig['swa_last']
    model_erm_swa.load_state_dict({k: v for k, v in model_erm_swa_dict.items()})
else:
    print('no swa_best checkpoint found in the ERM model_dict')

model_sam_dict_orig = torch.load('models/{}.pth'.format(args.model_path_sam))
model_sam_dict = model_sam_dict_orig['best'] if args.early_stopped_model_sam else model_sam_dict_orig['last']
model_sam.load_state_dict({k: v for k, v in model_sam_dict.items()})
if 'swa_best' in model_sam_dict_orig:
    model_sam_swa_dict = model_sam_dict_orig['swa_best']  # if args.early_stopped_model_sam else model_sam_dict_orig['swa_last']
    model_sam_swa.load_state_dict({k: v for k, v in model_sam_swa_dict.items()})
else:
    print('no swa_best checkpoint found in the SAM model_dict')

std_weight_perturb = 0.05
model_erm_plus_rand, model_sam_plus_rand = copy.deepcopy(model_erm), copy.deepcopy(model_sam)
utils_train.perturb_weights(model_erm_plus_rand, std_weight_perturb, 0, 'gauss')
utils_train.perturb_weights(model_sam_plus_rand, std_weight_perturb, 0, 'gauss')

model_zero = copy.deepcopy(model_erm)
utils_train.set_weights_to_zero(model_zero)

models_dict = {
    'erm': model_erm, 'erm_swa': model_erm_swa, 'sam': model_sam, 'sam_swa': model_sam_swa,
    'init': model_init1, 'erm_rand': model_erm_plus_rand, 'sam_rand': model_sam_plus_rand, 'zero': model_zero
}

train_batches_for_bn = data.get_loaders(args.dataset, 5000, 128, split='train', shuffle=True,
                                        data_augm=True, drop_last=False)  # needed to update the BN stats
train_batches_sharpness = data.get_loaders(args.dataset, args.n_eval_sharpness, args.bs_sharpness, split='train', shuffle=False,
                                           data_augm=False, drop_last=False)
eval_train_batches = data.get_loaders(args.dataset, args.n_eval, args.n_eval, split='train', shuffle=False,
                                      data_augm=False, drop_last=False)
eval_test_batches = data.get_loaders(args.dataset, args.n_eval, args.n_eval, split='test', shuffle=False,
                                     data_augm=False, drop_last=False)

# otherwise sharpness skyrockets on these models
if 'erm_rand' in [args.model_name1, args.model_name2]:
    utils_train.bn_update(train_batches_for_bn, model_erm_plus_rand)
if 'sam_rand' in [args.model_name1, args.model_name2]:
    utils_train.bn_update(train_batches_for_bn, model_sam_plus_rand)
# x_train, _, y_train, ln_train = [t.cuda() for t in list(eval_train_batches)[0]]
# x_test, _, y_test, ln_test = [t.cuda() for t in list(eval_test_batches)[0]]

err, _, _ = utils_eval.rob_err(eval_test_batches, model_erm, 0, 0, scaler, 0, 0)
print('ERM: err={:.2%}'.format(err))
err, _, _ = utils_eval.rob_err(eval_test_batches, model_erm_swa, 0, 0, scaler, 0, 0)
print('ERM + SWA: err={:.2%}'.format(err))
err, _, _ = utils_eval.rob_err(eval_test_batches, model_sam, 0, 0, scaler, 0, 0)
print('SAM: err={:.2%}'.format(err))
err, _, _ = utils_eval.rob_err(eval_test_batches, model_sam_swa, 0, 0, scaler, 0, 0)
print('SAM + SWA: err={:.2%}'.format(err))

print('dist(w_erm, w_rand)={:.3f}'.format(utils_eval.dist_models(model_erm, model_init1)))
print('dist(w_sam, w_rand)={:.3f}'.format(utils_eval.dist_models(model_sam, model_init2)))
print('dist(w_rand1, w_rand2)={:.3f}'.format(utils_eval.dist_models(model_init1, model_init2)))
print('dist(w_erm, w_sam)={:.3f}'.format(utils_eval.dist_models(model_erm, model_sam)))
print('dist(w_erm, w_erm_swa)={:.3f}'.format(utils_eval.dist_models(model_erm, model_erm_swa)))
print('dist(w_sam, w_erm_swa)={:.3f}'.format(utils_eval.dist_models(model_sam, model_erm_swa)))
print('||w_{} - w_{}||_2={:.2f}'.format(
    args.model_name1, args.model_name2, utils_eval.dist_models(models_dict[args.model_name1], models_dict[args.model_name2])))
print('||w_{}||_2={:.2f}, ||w_{}||_2={:.2f}'.format(
    args.model_name1, utils_eval.norm_weights(models_dict[args.model_name1]),
    args.model_name2, utils_eval.norm_weights(models_dict[args.model_name2])))

if args.distance_only:
    exit()

model1_sharpness_obj, model1_sharpness_grad_norm = utils_eval.eval_sharpness(
    models_dict[args.model_name1], train_batches_sharpness, loss_f, rho=0.1, n_steps=20, n_restarts=3)
model2_sharpness_obj, model2_sharpness_grad_norm = utils_eval.eval_sharpness(
    models_dict[args.model_name2], train_batches_sharpness, loss_f, rho=0.1, n_steps=20, n_restarts=3)
print('Model 1, sharpness: obj={:.2f}, grad_norm={:.2f}'.format(model1_sharpness_obj, model1_sharpness_grad_norm))
print('Model 2, sharpness: obj={:.2f}, grad_norm={:.2f}'.format(model2_sharpness_obj, model2_sharpness_grad_norm))

with torch.no_grad():
    alpha_step = 0.05  # 0.05 is sufficient
    alpha_range = np.concatenate([np.arange(-1.0, 0.0, alpha_step), np.arange(0.0, 2.0+alpha_step, alpha_step)])
    train_losses, test_losses = np.zeros_like(alpha_range), np.zeros_like(alpha_range)
    train_errors, test_errors = np.zeros_like(alpha_range), np.zeros_like(alpha_range)
    model1, model2 = models_dict[args.model_name1], models_dict[args.model_name2]
    for i, alpha in enumerate(alpha_range):
        for (p, p1, p2) in zip(model_interpolated.parameters(), model1.parameters(), model2.parameters()):
            p.data = (1 - alpha) * p1.data + alpha * p2.data  # alpha=0: first model, alpha=1: second model
        utils_train.bn_update(train_batches_for_bn, model_interpolated)

        train_errors[i], train_losses[i], _ = utils_eval.rob_err(eval_train_batches, model_interpolated, 0, 0, scaler, 0, 0)
        test_errors[i], test_losses[i], _ = utils_eval.rob_err(eval_test_batches, model_interpolated, 0, 0, scaler, 0, 0)
        print('alpha={:.2f}: loss={:.3}/{:.3}, err={:.2%}/{:.2%}'.format(
            alpha, train_losses[i], test_losses[i], train_errors[i], test_errors[i]))


export_dict = {'model_name1': args.model_name1, 'model_name2': args.model_name2,
               'alpha_range': alpha_range,
               'train_losses': train_losses, 'test_losses': test_losses,
               'train_errors': train_errors, 'test_errors': test_errors,
               'model1_norm': utils_eval.norm_weights(models_dict[args.model_name1]),
               'model2_norm': utils_eval.norm_weights(models_dict[args.model_name2]),
               'model1_sharpness_obj': model1_sharpness_obj, 'model2_sharpness_obj': model2_sharpness_obj,
               'model1_sharpness_grad_norm': model1_sharpness_grad_norm, 'model2_sharpness_grad_norm': model2_sharpness_grad_norm,
               }
np.save('metrics_loss_surface/dataset={} ln={} early_stopped_model={}-{} models={}-{} n_eval={} label={}.npy'.format(
    args.dataset, p_label_noise, args.early_stopped_model_erm, args.early_stopped_model_sam, args.model_name1,
    args.model_name2, args.n_eval, args.label),
    export_dict)
time_elapsed = time.time() - start_time
print('Done in {:.2f}m'.format((time.time() - start_time) / 60))

