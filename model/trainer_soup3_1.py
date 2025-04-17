import os
import numpy as np
import pandas as pd
import copy
import torch
from time import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import mean_absolute_error, roc_auc_score

from madani.utils.scaling import Scaler, DummyScaler
from madani.utils.optims_update import RobustL1, BCEWithLogitsLoss
from madani.utils.optims_update import Lamb, Lookahead
from madani.utils.utils import count_parameters, get_compute_device
from madani.utils.input_processing import DataHandler, data_type_torch


class ModelTrainer:
    def __init__(self,
                 model,
                 model_name='UnnamedModel',
                 n_elements=8,
                 capture_every=None,
                 verbose=True,
                 drop_unary=True,
                 scale=True):

        self.model = model
        self.model_name = model_name
        self.data_loader = None      
        self.train_loader = None     
        self.classification = False  
        self.n_elements = n_elements
        self.compute_device = model.compute_device
        self.fudge = 0.02  # random scaling factor for fraction "jitter"
        self.capture_every = capture_every
        self.verbose = verbose
        self.drop_unary = drop_unary
        self.scale = scale

        if self.compute_device is None:
            self.compute_device = get_compute_device()

        if self.verbose:
            print('\nModel architecture: out_dims, d_model, N, heads')
            print(f'{self.model.out_dims}, {self.model.d_model}, '
                  f'{self.model.N}, {self.model.heads}')
            print(f'Running on compute device: {self.compute_device}')
            print(f'Model size: {count_parameters(self.model)} parameters\n')
        if self.capture_every is not None:
            print(f'Capturing attention tensors every {self.capture_every}')

    def load_data(self, file_name, batch_size=2**9, train=False):
        self.batch_size = batch_size
        inference = not train
        data_loaders = DataHandler(file_name,
                                   batch_size=batch_size,
                                   n_elements=self.n_elements,
                                   inference=inference,
                                   verbose=self.verbose,
                                   drop_unary=self.drop_unary,
                                   scale=self.scale)

        print(f'Loading data with up to {data_loaders.n_elements:0.0f} elements in the formula')
        self.n_elements = data_loaders.n_elements
        data_loader = data_loaders.get_data_loaders(inference=inference)
        y = data_loader.dataset.y
        if train:
            self.train_len = len(y)
            self.scaler = DummyScaler(y) if self.classification else Scaler(y)
            self.train_loader = data_loader
        self.data_loader = data_loader

    def _train_phase(self, phase_name, epochs, checkin, optim_params):
        print(f"\n--- Starting {phase_name} phase for {epochs} epochs ---")
        self.criterion = RobustL1 if not self.classification else BCEWithLogitsLoss

        base_optim = Lamb(params=self.model.parameters())
        optimizer = base_optim
        self.optimizer = optimizer

        base_lr = optim_params.get('base_lr', 1e-4)
        max_lr = optim_params.get('max_lr', 6e-3)
        lr_decay = optim_params.get('lr_decay', 1.0)

        self.epochs_step = optim_params.get('epochs_step', 1)
        self.step_size = self.epochs_step * len(self.train_loader)
        self.lr_scheduler = CyclicLR(self.optimizer,
                                     base_lr=base_lr,
                                     max_lr=max_lr,
                                     cycle_momentum=False,
                                     step_size_up=self.step_size)
        self.stepping = True

        # Track best validation MAE
        best_val_mae = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            self.model.train()
            ti = time()
            for i, data in enumerate(self.train_loader):
                X, y, formula = data
                y = self.scaler.scale(y)
                src = X[:, :, 0]
                frac = X[:, :, 1]

                frac = frac * (1 + torch.randn_like(frac) * self.fudge)
                frac = torch.clamp(frac, 0, 1)
                frac[src == 0] = 0
                frac = frac / frac.sum(dim=1, keepdim=True)

                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)

                output = self.model.forward(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)
                loss = self.criterion(
                    prediction.reshape(-1),
                    uncertainty.reshape(-1),
                    y.reshape(-1)
                )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.stepping:
                    self.lr_scheduler.step()

            # Evaluate at check-in points
            if (epoch+1) % checkin == 0 or epoch == epochs - 1 or epoch == 0:
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                print(f"{phase_name} Epoch {epoch}/{epochs} --- val mae: {mae_v:0.4g}")

                # If improved, save this model state
                if mae_v < best_val_mae:
                    best_val_mae = mae_v
                    best_state = copy.deepcopy(self.model.state_dict())
                    print(f"New best {phase_name} model at epoch {epoch} with val MAE={mae_v:0.4g}")

            # Decay cyclic LR boundaries if requested
            if lr_decay < 1.0:
                new_base_lr = base_lr * (lr_decay ** (epoch+1))
                new_max_lr = max_lr * (lr_decay ** (epoch+1))
                self.lr_scheduler = CyclicLR(self.optimizer,
                                             base_lr=new_base_lr,
                                             max_lr=new_max_lr,
                                             cycle_momentum=False,
                                             step_size_up=self.step_size)

        # After all epochs, load the best model
        self.model.load_state_dict(best_state)
        print(f"Loaded best {phase_name} model with val MAE={best_val_mae:0.4g}")
        # Optionally save to disk
        self.save_network(model_name=f"{self.model_name}_{phase_name}_best")

    def pretrain(self, epochs=10, checkin=2, optim_params=None):
        if optim_params is None:
            optim_params = {'base_lr': 1e-4,'max_lr': 6e-3,'lr_decay': 1.0,'epochs_step': 1}
        print("\n=== Pretraining Phase ===")
        self._train_phase("Pretraining", epochs, checkin, optim_params)

    def finetune(self, epochs=20, checkin=2, optim_params=None):
        if optim_params is None:
            optim_params = {'base_lr': 1e-4,'max_lr': 6e-3,'lr_decay': 0.98,'epochs_step': 1}
        print("\n=== Fine-tuning Phase ===")
        self._train_phase("Fine-tuning", epochs, checkin, optim_params)
