from collections import defaultdict
from itertools import chain
from torch.optim import Optimizer
import torch
import warnings
import numpy as np

RNG_SEED = 42
torch.manual_seed(RNG_SEED)


# %%
class SWA(Optimizer):
    def __init__(self, optimizer, swa_start=None, swa_freq=None, swa_lr=None):
        r"""Implements Stochastic Weight Averaging (SWA).
        Stochastic Weight Averaging was proposed in `Averaging Weights Leads to
        Wider Optima and Better Generalization`_ by Pavel Izmailov, Dmitrii
        Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
        (UAI 2018).
        """
        self._auto_mode, (self.swa_start, self.swa_freq) = \
            self._check_params(self, swa_start, swa_freq)
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
                warnings.warn(
                    "Some of swa_start, swa_freq is None, ignoring swa_lr")
            # If not in auto mode make all swa parameters None
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
    def _check_params(self, swa_start, swa_freq):
        params = [swa_start, swa_freq]
        params_none = [param is None for param in params]
        if not all(params_none) and any(params_none):
            warnings.warn(
                "Some of swa_start, swa_freq is None, ignoring other")
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
        r"""Updates the SWA running averages for the given parameter group.
        Arguments:
            param_group (dict): Specifies for what parameter group SWA running
                averages should be updated
        Examples:
            >>> # automatic mode
            >>> base_opt = torch.optim.SGD([{'params': [x]},
            >>>             {'params': [y], 'lr': 1e-3}], lr=1e-2, momentum=0.9)
            >>> opt = torchcontrib.optim.SWA(base_opt)
            >>> for i in range(100):
            >>>     opt.zero_grad()
            >>>     loss_fn(model(input), target).backward()
            >>>     opt.step()
            >>>     if i > 10 and i % 5 == 0:
            >>>         # Update SWA for the second parameter group
            >>>         opt.update_swa_group(opt.param_groups[1])
            >>> opt.swap_swa_sgd()
        """
        if reset:
            # print('Resetting the swa_buffer')
            for p in group['params']:
                param_state = self.state[p]
                param_state['swa_buffer'] = torch.zeros_like(p.data)
            for group in self.param_groups:
                group['n_avg'] = 0
                group['step_counter'] = 0

        if group["n_avg"] >= 0:
            # print(f'Updating for weights with mae {mae:0.4e}')
            for p in group['params']:
                param_state = self.state[p]
                if 'swa_buffer' not in param_state:
                    param_state['swa_buffer'] = torch.zeros_like(p.data)
                buf = param_state['swa_buffer']
                virtual_decay = 1 / float(group["n_avg"] + 1)
                virtual_decay = (virtual_decay)**(1/3)
                diff = (p.data - buf) * virtual_decay
                buf.add_(diff)

        group["n_avg"] += 1


    def update_swa(self, mae):
        r"""Updates the SWA running averages of all optimized parameters.
        """
        self.maes.append(mae)
        self.mae_avg = np.mean(self.maes)
        self.minimum_found = False

        if mae < 0.98 * self.maximum_mae:
            # print(f'mae {mae:0.4e}, max {self.maximum_mae:0.4e}, '
            #       f'min {self.minimum_mae:0.4e}, resetting SWA, '
            #       f'updating maximum saved value')
            for group in self.param_groups:
                self.update_swa_group(group, reset=True, mae=mae)
            self.maximum_mae = mae

        if mae < self.minimum_mae:
            # print(f'{mae:0.4e} < {self.minimum_mae:0.4e}, '
            #       f'updating minimum saved value')
            self.minimum_mae = mae

        if mae <= 1.01 * self.minimum_mae:
            # print('Adding weights to SWA averaging')
            for group in self.param_groups:
                self.update_swa_group(group, mae=mae)
            self.minimum_found = True



    def swap_swa_sgd(self):
        r"""Swaps the values of the optimized variables and swa buffers.
        It's meant to be called in the end of training to use the collected
        swa running averages. It can also be used to evaluate the running
        averages during training; to continue training `swap_swa_sgd`
        should be called again.
        """
        err_swa_buffer = False
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'swa_buffer' not in param_state:
                    # If swa wasn't applied we don't swap params
                    err_swa_buffer = True
                    continue
                buf = param_state['swa_buffer']
                tmp = torch.empty_like(p.data)
                tmp.copy_(p.data)
                p.data.copy_(buf)
                buf.copy_(tmp)
        if err_swa_buffer:
            err_msg_swa_buffer = "No SWA weights to swap"
            warnings.warn(err_msg_swa_buffer)

    def step(self, closure=None):
        r"""Performs a single optimization step.
        In automatic mode also updates SWA running averages.
        """
        self._reset_lr_to_swa()
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            group["step_counter"] += 1
            steps = group["step_counter"]
            if self._auto_mode:
                if steps > self.swa_start and steps % self.swa_freq == 0:
                    self.update_swa_group(group)
        return loss

    def state_dict(self):
        r"""Returns the state of SWA as a :class:`dict`.
        It contains three entries:
            * opt_state - a dict holding current optimization state of the base
                optimizer. Its content differs between optimizer classes.
            * swa_state - a dict containing current state of SWA. For each
                optimized variable it contains swa_buffer keeping the running
                average of the variable
            * param_groups - a dict containing all parameter groups
        """
        opt_state_dict = self.optimizer.state_dict()
        swa_state = {(id(k) if isinstance(k, torch.Tensor) else k): v
                     for k, v in self.state.items()}
        opt_state = opt_state_dict["state"]
        param_groups = opt_state_dict["param_groups"]
        return {"opt_state": opt_state, "swa_state": swa_state,
                "param_groups": param_groups}

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.
        Args:
            state_dict (dict): SWA optimizer state. Should be an object returned
                from a call to `state_dict`.
        """
        swa_state_dict = {"state": state_dict["swa_state"],
                          "param_groups": state_dict["param_groups"]}
        opt_state_dict = {"state": state_dict["opt_state"],
                          "param_groups": state_dict["param_groups"]}
        super(SWA, self).load_state_dict(swa_state_dict)
        self.optimizer.load_state_dict(opt_state_dict)
        self.opt_state = self.optimizer.state

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.
        This can be useful when fine tuning a pre-trained network as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.
        Args:
            param_group (dict): Specifies what Tensors should be optimized along
            with group specific optimization options.
        """
        param_group['n_avg'] = 0
        param_group['step_counter'] = 0
        self.optimizer.add_param_group(param_group)

    @staticmethod
    def bn_update(loader, model, device=None):
        r"""Updates BatchNorm running_mean, running_var buffers in the model.
        It performs one pass over data in `loader` to estimate the activation
        statistics for BatchNorm layers in the model.
        Args:
            loader (torch.utils.data.DataLoader): dataset loader to compute the
                activation statistics on. Each data batch should be either a
                tensor, or a list/tuple whose first element is a tensor
                containing data.
            model (torch.nn.Module): model for which we seek to update BatchNorm
                statistics.
            device (torch.device, optional): If set, data will be trasferred to
                :attr:`device` before being passed into :attr:`model`.
        """
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


# BatchNorm utils
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
