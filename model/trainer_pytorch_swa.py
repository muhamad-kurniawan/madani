import os
import numpy as np
import pandas as pd
import torch
from time import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import mean_absolute_error, roc_auc_score

from madani.utils.scaling import Scaler, DummyScaler
from madani.utils.optims import RobustL1, BCEWithLogitsLoss
from madani.utils.optims import Lamb, Lookahead  # using your existing Lamb and Lookahead
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn  # native SWA utilities
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

        # Set device if not already specified
        if self.compute_device is None:
            self.compute_device = get_compute_device()

        # Internal flags/vars for attention capture etc.
        self.capture_flag = False
        self.formula_current = None
        self.act_v = None
        self.pred_v = None

        if self.verbose:
            print('\nModel architecture: out_dims, d_model, N, heads')
            print(f'{self.model.out_dims}, {self.model.d_model}, {self.model.N}, {self.model.heads}')
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
        # Update n_elements if it was adjusted during data handling
        self.n_elements = data_loaders.n_elements

        data_loader = data_loaders.get_data_loaders(inference=inference)
        y = data_loader.dataset.y
        if train:
            self.train_len = len(y)
            # Choose scaler for regression or classification
            if self.classification:
                self.scaler = DummyScaler(y)
            else:
                self.scaler = Scaler(y)
            self.train_loader = data_loader
        self.data_loader = data_loader

    def fit(self, epochs=None, checkin=None, losscurve=False, discard_n=10):
        # Safety checks
        assert self.train_loader is not None, "Please load training data."
        assert self.data_loader is not None, "Please load validation data."

        # Prepare tracking dictionaries for loss curves
        self.loss_curve = {'train': [], 'val': []}

        # Set how frequently to step the scheduler (here, one step per epoch)
        self.epochs_step = 1  
        self.step_size = self.epochs_step * len(self.train_loader)
        print(f'Stepping every {self.step_size} training passes, cycling LR every {self.epochs_step} epochs')

        # Default epochs if not specified
        if epochs is None:
            n_iterations = 1e4
            epochs = int(n_iterations / len(self.data_loader))
            print(f'Running for {epochs} epochs')

        # Check-in frequency for printing metrics
        if checkin is None:
            checkin = self.epochs_step * 2
            print(f'Checkin at {checkin} epochs to match LR scheduler')

        # Define loss function
        self.criterion = RobustL1
        if self.classification:
            print("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss

        # Build the base optimizer using Lamb and Lookahead
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = optimizer  # save for use in training

        # Set up learning rate schedulers:
        # Before SWA, use CosineAnnealingLR; during SWA phase, use SWALR.
        self.lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        self.swa_scheduler = SWALR(optimizer, swa_lr=0.05, anneal_epochs=5)

        # Set the epoch when SWA starts (adjust as needed)
        self.swa_start = 2  
        self.swa_model = None  # will be created at swa_start

        # Lists for tracking LR and SWA evaluation points
        self.lr_list = []
        self.xswa = []
        self.yswa = []

        print("Starting training...")
        for epoch in range(epochs):
            self.epoch = epoch
            self.epochs = epochs

            self.model.train()
            ti = time()
            for i, data in enumerate(self.train_loader):
                X, y, formula = data
                y = self.scaler.scale(y)

                src = X[:, :, 0]
                frac = X[:, :, 1]
                # Add noise ("jitter") to fractions for robustness
                frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)
                frac = torch.clamp(frac, 0, 1)
                frac[src == 0] = 0
                frac = frac / frac.sum(dim=1, keepdim=True)

                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)

                # Optional attention capture on each step
                if self.capture_every == 'step':
                    self.capture_flag = True
                    self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                    self.capture_flag = False

                # Forward pass
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

                # (Optional) If you wish to step the LR scheduler per batch, add that here.

            dt = time() - ti

            # SWA update: once we pass swa_start, update the SWA model and step SWALR.
            if epoch >= self.swa_start:
                if self.swa_model is None:
                    # Create SWA model using current weights
                    self.swa_model = AveragedModel(self.model)
                else:
                    self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.lr_scheduler.step()

            # Record and print metrics at check-in intervals
            if (epoch + 1) % checkin == 0 or epoch == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    act_t, pred_t, _, _ = self.predict(self.train_loader)
                mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve['train'].append(mae_t)

                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve['val'].append(mae_v)

                print(f'Epoch: {epoch}/{epochs} --- train mae: {mae_t:0.4g}  val mae: {mae_v:0.4g}')

                # Optionally track SWA evaluation points
                if epoch >= self.swa_start and (epoch + 1) % (2 * self.epochs_step) == 0:
                    self.xswa.append(epoch)
                    self.yswa.append(mae_v)

                # Optionally plot the loss curves
                if losscurve:
                    plt.figure(figsize=(8, 5))
                    xval = np.arange(len(self.loss_curve['val'])) * checkin
                    plt.plot(xval, self.loss_curve['train'], 'o-', label='train_mae')
                    plt.plot(xval, self.loss_curve['val'], 's--', label='val_mae')
                    plt.plot(self.xswa, self.yswa, 'o', ms=12, mfc='none', label='SWA point')
                    plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                    plt.title(f'{self.model_name}')
                    plt.xlabel('epochs')
                    plt.ylabel('MAE')
                    plt.legend()
                    plt.show()

            # (Optional) Add early stopping logic here if desired.

        # End-of-training: if SWA was used, update BatchNorm stats and swap to SWA model.
        if self.swa_model is not None:
            print("Updating BatchNorm statistics for SWA model...")
            update_bn(self.train_loader, self.swa_model, device=self.compute_device)
            # For final evaluation, replace current model with the SWA model.
            self.model = self.swa_model

    def predict(self, loader):
        """
        Runs inference on a given DataLoader.
        Returns (act, pred, formulae, uncert).
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
                    if isinstance(formula, (list, tuple)):
                        self.formula_current = list(formula)
                    else:
                        self.formula_current = [formula]
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
                data_loc = slice(i * self.batch_size, i * self.batch_size + len(y))
                atoms[data_loc, :] = src.cpu().numpy().astype('int32')
                fractions[data_loc, :] = frac.cpu().numpy().astype('float32')
                act[data_loc] = y.view(-1).cpu().numpy().astype('float32')
                pred[data_loc] = prediction.view(-1).cpu().numpy().astype('float32')
                uncert[data_loc] = uncertainty.view(-1).cpu().numpy().astype('float32')
                formulae[data_loc] = formula
        self.model.train()  # switch back to training mode if needed
        return (act, pred, formulae, uncert)

    def save_network(self, model_name=None):
        """
        Saves the model weights and scaler state.
        """
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
        """
        Loads model and scaler state from the .pth file.
        """
        path = f'models/trained_models/{path}'
        checkpoint = torch.load(path, map_location=self.compute_device)
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = optimizer  # reinitialize optimizer structure
        self.scaler = Scaler(torch.zeros(3))
        self.model.load_state_dict(checkpoint['weights'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        self.model_name = checkpoint['model_name']
        print(f"Loaded network '{self.model_name}' from {path}")
