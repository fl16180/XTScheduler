import re
from collections import defaultdict
import math

import torch
from torch.optim.lr_scheduler import _LRScheduler


def extract_x_from_param(par_name, x_template):
    pattern = re.compile(x_template)
    match = pattern.search(par_name)
    if match:
        return int(match.group(1))
    return None


class BaseXTScheduler(_LRScheduler):
    def __init__(self, model, optimizer_class, optimizer_params,
                 x_template='[a-z]+(\d+).', last_epoch=-1, verbose=False):
        
        # make mapping from x to param (name, group) tuple
        self.param_x_groups = self._parse_params_in_space(model, x_template)
        self.nx = len(self.param_x_groups)

        # mapping with only param without names
        x_to_group = self.get_group_params()
        optim_param_groups = []
        for x, group in x_to_group.items():
            param_dict = {'params': group}
            optim_param_groups.append(param_dict)

        optimizer = optimizer_class(optim_param_groups, **optimizer_params)

        super().__init__(optimizer, last_epoch, verbose)

    def _parse_params_in_space(self, model, x_template):
        param_x_groups = defaultdict(list)
        orphan_params = []
        default_x = -1
        for par_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            _x = extract_x_from_param(par_name, x_template)
            if _x is not None:
                param_x_groups[_x].append((par_name, param))
                default_x = _x
            else:
                if default_x == -1:
                    orphan_params.append((par_name, param))
                else:
                    param_x_groups[default_x].append((par_name, param))

        # our convention is that layer numbering starts at x=0
        assert min(param_x_groups.keys()) in [0, 1]
        if min(param_x_groups.keys()) == 1:
            shifted_param_groups = {}
            for x, group in param_x_groups.items():
                shifted_param_groups[x - 1] = group
            param_x_groups = shifted_param_groups

        # orphan params assigned to the beginning
        if len(orphan_params) > 0:
            param_x_groups[0] = orphan_params + param_x_groups[0]

        return param_x_groups

    def get_group_par_names(self):
        out = {}
        for x, group in self.param_x_groups.items():
            name_group = [x[0] for x in group]
            out[x] = name_group
        return out

    def get_group_params(self):
        out = {}
        for x, group in self.param_x_groups.items():
            par_group = [x[1] for x in group]
            out[x] = par_group
        return out

    def set_xt_params(self, **kwargs):
        raise NotImplementedError

    def get_lr(self):
        raise NotImplementedError

    @property
    def optimizer(self):
        return self.optimizer


class LambdaXTScheduler(BaseXTScheduler):
    def __init__(self, model, optimizer_class, optimizer_params,
                 xt_func=None, x_func=None, t_func=None,
                 x_template='[a-z]+(\d+).', last_epoch=-1, verbose=False):

        self._set_xt_params(xt_func, x_func, t_func)
        super().__init__(model, optimizer_class, optimizer_params,
                         x_template, last_epoch, verbose)

    def _set_xt_params(self, xt_func=None, x_func=None, t_func=None):
        if xt_func is not None:
            self.xt_func = xt_func
        else:
            assert x_func is not None and t_func is not None
            def xt_func(x, t):
                return x_func(x) * t_func(t)
            self.xt_func = xt_func

    def get_lr(self):
        t = self.last_epoch
        return [self.xt_func(x, t) for x in range(self.nx)]


class CosineXTScheduler(BaseXTScheduler):
    def __init__(self, model, optimizer_class, optimizer_params,
                 nx, nt, lr_min=1e-4, lr_max=1e-1, x_pulses=0.5, t_pulses=3,
                 x_template='[a-z]+(\d+).', last_epoch=-1, verbose=False):

        self._set_xt_params(nx, nt, lr_min, lr_max, x_pulses, t_pulses)
        super().__init__(model, optimizer_class, optimizer_params,
                         x_template, last_epoch, verbose)

    def _set_xt_params(self, nx, nt, lr_min, lr_max, x_pulses=0.5, t_pulses=3):
        xt_func = self._suggest_joint_cosine(nx, nt, lr_min, lr_max, x_pulses, t_pulses)
        self.xt_func = xt_func

    def _suggest_joint_cosine(self, nx, nt, lr_min, lr_max, x_pulses=0.5, t_pulses=3): 
        per_x = nx / x_pulses       # pulses over x (default to half period)
        per_t = nt / t_pulses       # pulses over t
        
        c = 2 * math.pi

        def joint_func(x, t):
            return (lr_max - lr_min) * (math.cos(c * (x / per_x + t / per_t)) + 1) / 2 + lr_min

        return joint_func

    def get_lr(self):
        t = self.last_epoch
        return [self.xt_func(x, t) for x in range(self.nx)]
