import os
import copy
import numpy as np
import pandas as pd
import torch
from time import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import mean_absolute_error, roc_auc_score

from madani.utils.scaling import Scaler, DummyScaler
from madani.utils.optims import RobustL1, BCEWithLogitsLoss
from madani.utils.optims import Lamb, Lookahead
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

        # Set device if not already set
        if self.compute_device is None:
            self.compute_device = get_compute_device()

        # Internal flags/vars
        self.capture_flag = False
        self.formula_current = None
        self.act_v = None
        self.pred_v = None

        # For Auto-Ensemble snapshots
        self.snapshots = []

        # For recording checkin points for plotting loss curves
        self.xswa = []
        self.yswa = []

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

        print(f'Loading data with up to {data_loaders.n_elements:0.0f} '
              f'elements in the formula')
        self.n_elements = data_loaders.n_elements

        data_loader = data_loaders.get_data_loaders(inference=inference)
        y = data_loader.dataset.y
        if train:
            self.train_len = len(y)
            if self.classification:
                self.scaler = DummyScaler(y)
            else:
                self.scaler = Scaler(y)
            self.train_loader = data_loader
        self.data_loader = data_loader

    def fit(self, epochs=None, checkin=None, losscurve=False, 
            adaptive_threshold=0.01, pretrain_epochs=0):
        """
        Trains the model using a cyclic LR scheduler.
        
        pretrain_epochs: If > 0, run a pre-training phase for these many epochs
                         before starting the main training with adaptive snapshot capture.
                         
        At every checkin interval in the main training phase, the validation MAE is computed.
        If the relative improvement over the best validation MAE is less than adaptive_threshold,
        a snapshot of the model is captured.
        """
        assert self.train_loader is not None, "Please load training data."
        assert self.data_loader is not None, "Please load validation data."

        # Initialize loss curves
        self.loss_curve = {'train': [], 'val': []}
        self.epochs_step = 1  # step after every epoch
        self.step_size = self.epochs_step * len(self.train_loader)
        print(f'Stepping every {self.step_size} training passes, cycling LR every {self.epochs_step} epochs')

        # Determine total epochs if not specified (for main training phase)
        if epochs is None:
            n_iterations = 1e4
            epochs = int(n_iterations / len(self.data_loader))
            print(f'Running for {epochs} epochs (main training phase)')

        if checkin is None:
            checkin = self.epochs_step * 2
            print(f'Checkin at {checkin} epochs to match LR scheduler')

        # --- PRE-TRAINING PHASE ---
        if pretrain_epochs > 0:
            print(f"Starting pre-training phase for {pretrain_epochs} epochs...")
            # We use the same optimizer and LR scheduler here.
            base_optim = Lamb(params=self.model.parameters())
            self.optimizer = Lookahead(base_optimizer=base_optim)
            pretrain_scheduler = CyclicLR(self.optimizer,
                                          base_lr=1e-4,
                                          max_lr=6e-3,
                                          cycle_momentum=False,
                                          step_size_up=self.step_size)
            for epoch in range(pretrain_epochs):
                self.model.train()
                ti = time()
                for i, data in enumerate(self.train_loader):
                    X, y, formula = data
                    y = self.scaler.scale(y)

                    src = X[:, :, 0]
                    frac = X[:, :, 1]
                    frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)
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

                    pretrain_scheduler.step()

                dt = time() - ti
                print(f"Pre-training Epoch {epoch+1}/{pretrain_epochs} complete in {dt:0.2f}s")
            print("Pre-training phase completed.\n")

            # Optionally, reinitialize the LR scheduler for the main training phase.
            self.lr_scheduler = CyclicLR(self.optimizer,
                                         base_lr=1e-4,
                                         max_lr=6e-3,
                                         cycle_momentum=False,
                                         step_size_up=self.step_size)
        else:
            # If no pre-training phase, initialize optimizer and scheduler now.
            base_optim = Lamb(params=self.model.parameters())
            self.optimizer = Lookahead(base_optimizer=base_optim)
            self.lr_scheduler = CyclicLR(self.optimizer,
                                         base_lr=1e-4,
                                         max_lr=6e-3,
                                         cycle_momentum=False,
                                         step_size_up=self.step_size)

        # --- MAIN TRAINING PHASE WITH AUTO-ENSEMBLE ---
        self.best_val = float('inf')
        self.lr_list = []  # reset LR list
        for epoch in range(epochs):
            self.epoch = epoch
            self.model.train()
            ti = time()

            for i, data in enumerate(self.train_loader):
                X, y, formula = data
                y = self.scaler.scale(y)

                src = X[:, :, 0]
                frac = X[:, :, 1]
                frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)
                frac = torch.clamp(frac, 0, 1)
                frac[src == 0] = 0
                frac = frac / frac.sum(dim=1, keepdim=True)

                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)

                if self.capture_every == 'step':
                    self.capture_flag = True
                    self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                    self.capture_flag = False

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

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

            dt = time() - ti

            if self.capture_every == 'epoch':
                self.capture_flag = True
                self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                self.capture_flag = False

            self.lr_list.append(self.optimizer.param_groups[0]['lr'])

            if (epoch+1) % checkin == 0 or epoch == epochs - 1 or epoch == 0:
                with torch.no_grad():
                    act_t, pred_t, _, _ = self.predict(self.train_loader)
                mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve['train'].append(mae_t)

                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve['val'].append(mae_v)

                epoch_str = f'Epoch: {epoch}/{epochs} ---'
                train_str = f'train mae: {mae_t:0.4g}'
                val_str = f'val mae: {mae_v:0.4g}'
                if self.classification:
                    train_auc = roc_auc_score(act_t, pred_t)
                    val_auc = roc_auc_score(act_v, pred_v)
                    train_str = f'train auc: {train_auc:0.3f}'
                    val_str = f'val auc: {val_auc:0.3f}'

                print(epoch_str, train_str, val_str)

                self.xswa.append(epoch)
                self.yswa.append(mae_v)

                # Adaptive snapshot capture:
                if self.best_val == float('inf'):
                    self.best_val = mae_v
                else:
                    improvement = self.best_val - mae_v
                    if improvement < self.best_val * adaptive_threshold:
                        snapshot = copy.deepcopy(self.model.state_dict())
                        self.snapshots.append(snapshot)
                        print(f"Auto-Ensemble: Captured snapshot at epoch {epoch+1} "
                              f"(improvement: {improvement:0.4f}). Total snapshots: {len(self.snapshots)}")
                    else:
                        self.best_val = mae_v

                if losscurve:
                    plt.figure(figsize=(8, 5))
                    xval = np.arange(len(self.loss_curve['val'])) * checkin
                    plt.plot(xval, self.loss_curve['train'], 'o-', label='train_mae')
                    plt.plot(xval, self.loss_curve['val'], 's--', label='val_mae')
                    plt.plot(np.array(self.xswa), np.array(self.yswa),
                             'o', ms=12, mfc='none', label='checkin point')
                    plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                    plt.title(f'{self.model_name}')
                    plt.xlabel('epochs')
                    plt.ylabel('MAE')
                    plt.legend()
                    plt.show()

        print(f"Training finished. Total snapshots captured: {len(self.snapshots)}")

    def predict(self, loader):
        """
        Runs inference on a given DataLoader.
        Returns: (act, pred, formulae, uncert)
        """
        if isinstance(loader, str):
            loader = DataHandler(loader,
                                 batch_size=64,
                                 n_elements=self.n_elements,
                                 inference=True,
                                 verbose=self.verbose,
                                 drop_unary=self.drop_unary,
                                 scale=self.scale)
        len_dataset = len(loader.dataset)
        n_atoms = int(len(loader.dataset[0][0]))

        act = np.zeros(len_dataset)
        pred = np.zeros(len_dataset)
        uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=object)

        atoms = np.empty((len_dataset, n_atoms))
        fractions = np.empty((len_dataset, n_atoms))

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula = data
                if self.capture_flag:
                    self.formula_current = list(formula) if isinstance(formula, (tuple, list)) else formula

                src = X[:, :, 0]
                frac = X[:, :, 1]
                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)

                output = self.model.forward(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)

                uncertainty = torch.exp(uncertainty) * self.scaler.std
                prediction = self.scaler.unscale(prediction)
                if self.classification:
                    prediction = torch.sigmoid(prediction)

                data_loc = slice(i*self.batch_size, i*self.batch_size + len(y))
                atoms[data_loc, :] = src.cpu().numpy().astype('int32')
                fractions[data_loc, :] = frac.cpu().numpy().astype('float32')
                act[data_loc] = y.view(-1).cpu().numpy().astype('float32')
                pred[data_loc] = prediction.view(-1).cpu().numpy().astype('float32')
                uncert[data_loc] = uncertainty.view(-1).cpu().numpy().astype('float32')
                formulae[data_loc] = formula

        self.model.train()
        return (act, pred, formulae, uncert)

    def save_network(self, model_name=None):
        if model_name is None:
            model_name = self.model_name
        os.makedirs('models/trained_models', exist_ok=True)
        path = f'models/trained_models/{model_name}.pth'
        print(f'Saving network ({model_name}) to {path}')

        save_dict = {
            'weights': self.model.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'model_name': model_name
        }
        torch.save(save_dict, path)

    def load_network(self, path):
        path = f'models/trained_models/{path}'
        checkpoint = torch.load(path, map_location=self.compute_device)

        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = optimizer

        self.scaler = Scaler(torch.zeros(3))

        self.model.load_state_dict(checkpoint['weights'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        self.model_name = checkpoint['model_name']
        print(f"Loaded network '{self.model_name}' from {path}")
