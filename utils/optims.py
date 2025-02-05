from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings
import numpy as np

RNG_SEED = 42
torch.manual_seed(RNG_SEED)


# %%
import torch
import warnings
from collections import defaultdict
from torch.optim.optimizer import Optimizer

class SWA(Optimizer):
    def __init__(self, optimizer, swa_start=None, swa_freq=None, swa_lr=None):
        r"""Implements Stochastic Weight Averaging (SWA).

        SWA was proposed in "Averaging Weights Leads to Wider Optima and Better Generalization"
        by Izmailov et al. (UAI 2018).
        """
        # Determine whether we are in "auto mode" (both swa_start and swa_freq provided)
        self._auto_mode, (self.swa_start, self.swa_freq) = self._check_params(swa_start, swa_freq)
        self.swa_lr = swa_lr
        self.minimum_mae = 1e9
        self.maximum_mae = 1e9
        self.maes = []
        self.discard_count = 0

        if self._auto_mode:
            if swa_start < 0:
                raise ValueError(f"Invalid swa_start: {swa_start}")
            if swa_freq < 1:
                raise ValueError(f"Invalid swa_freq: {swa_freq}")
        else:
            if self.swa_lr is not None:
                warnings.warn("Some of swa_start, swa_freq is None, ignoring swa_lr")
            self.swa_lr = None
            self.swa_start = None
            self.swa_freq = None

        if self.swa_lr is not None and self.swa_lr < 0:
            raise ValueError(f"Invalid SWA learning rate: {swa_lr}")

        self.optimizer = optimizer
        self.defaults = self.optimizer.defaults
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.opt_state = self.optimizer.state
        for group in self.param_groups:
            group['n_avg'] = 0
            group['step_counter'] = 0

    @staticmethod
    def _check_params(swa_start, swa_freq):
        params = [swa_start, swa_freq]
        params_none = [param is None for param in params]
        if not all(params_none) and any(params_none):
            warnings.warn("Some of swa_start, swa_freq is None, ignoring other")
        for i, param in enumerate(params):
            if param is not None and not isinstance(param, int):
                params[i] = int(param)
                warnings.warn("Casting swa_start, swa_freq to int")
        return not any(params_none), params

    def _reset_lr_to_swa(self):
        if self.swa_lr is None:
            return
        for param_group in self.param_groups:
            if param_group['step_counter'] >= self.swa_start:
                param_group['lr'] = self.swa_lr

    def update_swa_group(self, group, reset=False, mae=None):
        if reset:
            for p in group['params']:
                param_state = self.state[p]
                param_state['swa_buffer'] = torch.zeros_like(p.data)
            group['n_avg'] = 0
            group['step_counter'] = 0

        # Update SWA for this group
        for p in group['params']:
            param_state = self.state[p]
            if 'swa_buffer' not in param_state:
                param_state['swa_buffer'] = torch.zeros_like(p.data)
            buf = param_state['swa_buffer']
            virtual_decay = 1 / float(group["n_avg"] + 1)
            virtual_decay = virtual_decay ** (1 / 3)
            diff = (p.data - buf) * virtual_decay
            buf.add_(diff)
        group["n_avg"] += 1

    def update_swa(self, mae):
        self.maes.append(mae)
        self.mae_avg = np.mean(self.maes)
        self.minimum_found = False

        if mae < 0.98 * self.maximum_mae:
            for group in self.param_groups:
                self.update_swa_group(group, reset=True, mae=mae)
            self.maximum_mae = mae

        if mae < self.minimum_mae:
            self.minimum_mae = mae

        if mae <= 1.01 * self.minimum_mae:
            for group in self.param_groups:
                self.update_swa_group(group, mae=mae)
            self.minimum_found = True

    def swap_swa_sgd(self):
        err_swa_buffer = False
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'swa_buffer' not in param_state:
                    err_swa_buffer = True
                    continue
                buf = param_state['swa_buffer']
                tmp = torch.empty_like(p.data)
                tmp.copy_(p.data)
                p.data.copy_(buf)
                buf.copy_(tmp)
        if err_swa_buffer:
            warnings.warn("No SWA weights to swap")

    def step(self, closure=None):
        self._reset_lr_to_swa()
        loss = self.optimizer.step(closure)
        # Increment step counters, but DO NOT automatically update SWA here.
        for group in self.param_groups:
            group["step_counter"] += 1
        return loss

    # state_dict, load_state_dict, add_param_group and bn_update remain unchanged.
    def state_dict(self):
        opt_state_dict = self.optimizer.state_dict()
        swa_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                     for k, v in self.state.items()}
        opt_state = opt_state_dict["state"]
        param_groups = opt_state_dict["param_groups"]
        return {"opt_state": opt_state, "swa_state": swa_state,
                "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        swa_state_dict = {"state": state_dict["swa_state"],
                          "param_groups": state_dict["param_groups"]}
        opt_state_dict = {"state": state_dict["opt_state"],
                          "param_groups": state_dict["param_groups"]}
        super(SWA, self).load_state_dict(swa_state_dict)
        self.optimizer.load_state_dict(opt_state_dict)
        self.opt_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group['n_avg'] = 0
        param_group['step_counter'] = 0
        self.optimizer.add_param_group(param_group)

    @staticmethod
    def bn_update(loader, model, device=None):
        if not _check_bn(model):
            return
        was_training = model.training
        model.train()
        momenta = {}
        model.apply(_reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        for input in loader:
            if isinstance(input, (list, tuple)):
                input = input[0]
            b = input.size(0)
            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum
            if device is not None:
                input = input.to(device)
            model(input)
            n += b
        model.apply(lambda module: _set_momenta(module, momenta))
        model.train(was_training)


# BatchNorm utility functions remain the same:
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True

def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]

def _reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)

def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum

def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


class Lamb(Optimizer):
    r"""Implements Lamb algorithm.
    It has been proposed in `Large Batch Optimization for Deep Learning:
        Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this
            into Adam. Useful for comparison purposes.
        _Large Batch Optimization for Deep Learning: Training BERT in 76
            minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-6,
                 weight_decay=0,
                 adam=False,
                 min_trust=None):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if min_trust and not 0.0 <= min_trust < 1.0:
            raise ValueError(f"Minimum trust range from 0 to 1: {min_trust}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        self.min_trust = min_trust
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    err_msg = "Lamb does not support sparse gradients, " + \
                        "consider SparseAdam instad."
                    raise RuntimeError(err_msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_((1 - beta1) * grad)
                # v_t
                # exp_avg_sq.mul_(beta2).addcmul_((1 - beta2) * grad *
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                # Paper v3 does not use debiasing.
                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # Apply bias to lr to avoid broadcast.
                step_size = group[
                    "lr"
                ]  # * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(group["weight_decay"], p.data)

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                if self.min_trust:
                    trust_ratio = max(trust_ratio, self.min_trust)
                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(-step_size * trust_ratio * adam_step)

        return loss


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid lookahead steps: {k}")
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if "slow_buffer" not in param_state:
                param_state["slow_buffer"] = torch.empty_like(fast_p.data)
                param_state["slow_buffer"].copy_(fast_p.data)
            slow = param_state["slow_buffer"]
            slow.add_(group["lookahead_alpha"] * (fast_p.data - slow))
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group["lookahead_step"] += 1
            if group["lookahead_step"] % group["lookahead_k"] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            "state": state_dict["state"],
            "param_groups": state_dict["param_groups"],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if "slow_state" not in state_dict:
            print("Loading state_dict from optimizer without Lookahead applied.")
            state_dict["slow_state"] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict[
                "param_groups"
            ],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = (
            self.base_optimizer.param_groups
        )  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)

def BCEWithLogitsLoss(output, log_std, target):
    loss = nn.functional.binary_cross_entropy_with_logits(output, target)
    return loss


def RobustL1(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    absolute = torch.abs(output - target)
    loss = np.sqrt(2.0) * absolute * torch.exp(-log_std) + log_std
    return torch.mean(loss)


def RobustL2(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    squared = torch.pow(output - target, 2.0)
    loss = 0.5 * squared * torch.exp(-2.0 * log_std) + log_std
    return torch.mean(loss)
