import torch
import utils
import copy


class SAM(torch.optim.Optimizer):
    """
    Implementation of SAM is based on https://github.com/davda54/sam/blob/main/sam.py.
    """
    def __init__(self, params, base_optimizer, rho, sam_no_grad_norm, only_sam_step_size=False, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.sam_no_grad_norm = sam_no_grad_norm
        # `only_sam_step_size`: we modify the step size (grad(w_sam) / grad(w) multiplicative factor) but keep the
        #                       original gradient direction as in the point `w`.
        self.only_sam_step_size = only_sam_step_size
        self.grad_norm_w, self.grad_norm_w_sam = None, None
        self.grad_w = dict()

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    @torch.no_grad()
    def first_step(self):
        """
        At the beginning of `first_step()`, the grads are at point `w`.
        Then this method updates the main model parameters in opt.param_groups[0]['params']
        """
        grad_norm = 1 if self.sam_no_grad_norm else self._grad_norm()
        self.grad_norm_w = self._grad_norm()
        # print(self.grad_norm_w)
        for group in self.param_groups:  # by default, there is only 1 param group
            # standard SGD optimizer contains the following keys:
            # ['params', 'lr', 'momentum', 'dampening', 'weight_decay', 'nesterov']
            scale = group["rho"] / (grad_norm + 1e-12)

            for param in group["params"]:  # group["params"] is a list with 56 params
                if param.grad is None: continue
                delta_w = scale * param.grad
                param.add_(delta_w)  # climb to the local maximum "w + e(w)"
                # by default, opt.state==defaultdict(<class 'dict'>, {}) but we can use it to store the SAM's delta
                self.state[param]["delta_w"] = delta_w  # only the ref to the `param.data` is used as the key
                self.grad_w[param] = param.grad.clone()  # store it to apply on 2nd step (if only_sam_step_size==True)

        self.zero_grad()  # and we zero out the first grads (since we've already stored them)

    @torch.no_grad()
    def second_step(self):
        """
        At the beginning of `second_step()`, the grads are at point `w + delta`.
        """
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None: continue
                param.sub_(self.state[param]["delta_w"])  # get back to `w` from `w + delta`

        if self.only_sam_step_size:  # put the original gradient and change only the step size
            self.grad_norm_w_sam = self._grad_norm()
            for group in self.param_groups:
                for param in group['params']:
                    param.grad = self.grad_w[param] * self.grad_norm_w_sam / self.grad_norm_w
                    # param.grad = param.grad * self.grad_norm_w / self.grad_norm_w_sam

    @torch.no_grad()
    def step(self, closure=None):
        self.base_optimizer.step()


def zero_init_delta_dict(delta_dict, rho):
    for param in delta_dict:
        delta_dict[param] = torch.zeros_like(param).cuda()

    delta_norm = torch.cat([delta_param.flatten() for delta_param in delta_dict.values()]).norm()
    for param in delta_dict:
        delta_dict[param] *= rho / delta_norm

    return delta_dict

def random_init_on_sphere_delta_dict(delta_dict, rho):
    for param in delta_dict:
        delta_dict[param] = torch.randn_like(param).cuda()

    delta_norm = torch.cat([delta_param.flatten() for delta_param in delta_dict.values()]).norm()
    for param in delta_dict:
        delta_dict[param] *= rho / delta_norm

    return delta_dict


def weight_ascent_step(model, f, orig_param_dict, delta_dict, step_size, rho, layer_name_pattern='all', no_grad_norm=False, verbose=False):
    utils.zero_grad(model)
    obj = f(model)  # grads are accumulated
    obj.backward()

    grad_norm = utils.get_flat_grad(model).norm()
    if verbose:
        print('obj={:.3f}, grad_norm={:.3f}'.format(obj, grad_norm))

    for param_name, param in model.named_parameters():
        if layer_name_pattern == 'all' or layer_name_pattern in param_name:
            if no_grad_norm:
                delta_dict[param] += step_size * param.grad
            else:
                delta_dict[param] += step_size / (grad_norm + 1e-12) * param.grad

    delta_norm = torch.cat([delta_param.flatten() for delta_param in delta_dict.values()]).norm()
    if delta_norm > rho:
        for param in delta_dict:
            delta_dict[param] *= rho / delta_norm

    # now apply the (potentially) scaled perturbation to modify the weight
    for param in model.parameters():
        param.data = orig_param_dict[param] + delta_dict[param]

    utils.zero_grad(model)
    return delta_dict


def perturb_weights_sam(model, f, rho, step_size, n_iters, no_grad_norm, rand_init=False, verbose=False):
    delta_dict = {param: torch.zeros_like(param) for param in model.parameters()}

    # random init on the sphere of radius `rho`
    if rand_init:
        delta_dict = random_init_on_sphere_delta_dict(delta_dict, rho)
        for param in model.parameters():
            param.data += delta_dict[param]
    
    orig_param_dict = {param: param.clone() for param in model.parameters()}

    for iter in range(n_iters):
        delta_dict = weight_ascent_step(model, f, orig_param_dict, delta_dict, step_size, rho, no_grad_norm=no_grad_norm, verbose=False)
    
    return delta_dict

