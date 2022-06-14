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
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar100', 'cifar10_horse_car', 'cifar10_dog_cat', 'uniform_noise'], type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'resnet18_gn', 'resnet34', 'resnet34_gn'], type=str)
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--model_type', type=str, choices=['last', 'best', 'swa_last', 'swa_best'], default='last', help='model type: last/best vs standard/swa')
    parser.add_argument('--swa_model', action='store_true', help='eval the model obtained via SWA')
    parser.add_argument('--layer_name_pattern', default='all', type=str, help='which layers to perturb (`all` means all learnable layers)')
    parser.add_argument('--n_eval', default=1024, type=int, help='#examples to evaluate on error')
    parser.add_argument('--bs', default=128, type=int, help='batch size for error computation')
    parser.add_argument('--n_eval_sharpness', default=256, type=int, help='#examples to evaluate on sharpness')
    parser.add_argument('--bs_sharpness', default=128, type=int, help='batch size for sharpness experiments')
    parser.add_argument('--rho', default=0.1, type=float, help='L2 radius for sharpness')
    parser.add_argument('--scale_last_layer', default=1.0, type=float, help='multiplicative coefficient to scale the last weight layer')
    parser.add_argument('--step_size_mult', default=0.1, type=float, help='step size multiplier for sharpness')
    parser.add_argument('--n_iters', default=20, type=int, help='number of iterations for sharpness')
    parser.add_argument('--n_restarts', default=1, type=int, help='number of restarts for sharpness')
    parser.add_argument('--loss', default='ce', choices=['ce', 'ce_offset', 'gce', 'smallest_k_ce'], type=str, help='Loss type.')
    parser.add_argument('--p_label_noise', default=0.0, type=float, help='label noise for evaluation')
    parser.add_argument('--model_width', default=64, type=int, help='model width (# conv filters on the first layer for ResNets)')
    parser.add_argument('--sharpness_on_test_set', action='store_true', help='compute sharpness on the test set')
    parser.add_argument('--sharpness_rand_init', action='store_true', help='random initialization')
    parser.add_argument('--merge_bn_stats', action='store_true', help='merge BN means and variances to its learnable parameters')
    parser.add_argument('--no_grad_norm', action='store_true', help='no gradient normalization in PGA')
    parser.add_argument('--half_prec', action='store_true', help='eval in half precision')
    parser.add_argument('--apply_step_size_schedule', action='store_true', help='apply step size schedule')
    parser.add_argument('--batch_transfer', action='store_true', help='batch transfer')
    parser.add_argument('--random_targets', action='store_true', help='random targets in the loss')
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()


start_time = time.time()
args = get_args()
print_stats = False
n_cls = 2 if args.dataset in ['cifar10_horse_car', 'cifar10_dog_cat'] else 10 if args.dataset != 'cifar100' else 100
p_label_noise = args.model_path.split('p_label_noise=')[1].split(' ')[0]
sharpness_split = 'test' if args.sharpness_on_test_set else 'train'
assert p_label_noise == args.model_path.split('p_label_noise=')[1].split(' ')[0], 'ln level should be the same for the visualization'
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

model = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width).cuda().eval()

model_dict = torch.load('models/{}.pth'.format(args.model_path))[args.model_type]

model_dict['linear.weight'] *= args.scale_last_layer 
model_dict['linear.bias'] *= args.scale_last_layer 
# for key in model_dict.keys():
#     if 'running_mean' not in key and 'running_var' not in key and 'num_batches_tracked' not in key:
#         model_dict[key] *= args.scale_last_layer

if args.merge_bn_stats:
    model_dict = utils_eval.bn_merge_means_variances(model_dict)

model.load_state_dict({k: v for k, v in model_dict.items()})  
# model.apply(models.init_weights(args.model))  # for random init
print('Weight norm: {:.3f}'.format(utils_eval.norm_weights(model)))

# important to exclude the validation samples to get the correct training error
n_val = int(0.1 * data.shapes_dict[args.dataset][0])
val_indices = np.random.permutation(data.shapes_dict[args.dataset][0])[:n_val]

eval_train_batches = data.get_loaders(args.dataset, args.n_eval, args.n_eval, split='train', shuffle=False,
                                      data_augm=False, drop_last=False, val_indices=val_indices)
eval_test_batches = data.get_loaders(args.dataset, args.n_eval, args.n_eval, split='test', shuffle=False,
                                     data_augm=False, drop_last=False)
train_err, train_loss, _ = utils_eval.rob_err(eval_train_batches, model, 0, 0, scaler, 0, 0)
test_err, test_loss, _ = utils_eval.rob_err(eval_test_batches, model, 0, 0, scaler, 0, 0)
print('[train] err={:.2%} loss={:.4f}, [test] err={:.2%}, loss={:.4f}'.format(train_err, train_loss, test_err, test_loss))

step_size = args.step_size_mult * args.rho
batches_sharpness = data.get_loaders(args.dataset, args.n_eval_sharpness, args.bs_sharpness, split=sharpness_split, shuffle=False,
                                     data_augm=False, drop_last=False, val_indices=val_indices)
sharpness_obj, sharpness_err, sharpness_grad_norm = utils_eval.eval_sharpness(
    model, batches_sharpness, loss_f, args.rho, step_size, args.n_iters, args.n_restarts, args.apply_step_size_schedule, args.no_grad_norm, 
    args.layer_name_pattern, args.random_targets, args.batch_transfer, rand_init=args.sharpness_rand_init, verbose=True)
print('sharpness: obj={:.5f}, err={:.2%}'.format(sharpness_obj, sharpness_err))

print('Done in {:.2f}m'.format((time.time() - start_time) / 60))

