import os
import numpy as np
import pandas as pd
import torch
from time import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import mean_absolute_error, roc_auc_score

from madani.utils.scaling import Scaler, DummyScaler
from madani.utils.optims import RobustL1, BCEWithLogitsLoss
from madani.utils.optims import Lamb, Lookahead, SWA
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

        # Set device
        if self.compute_device is None:
            self.compute_device = get_compute_device()

        # Internal flags/vars
        self.capture_flag = False
        self.formula_current = None
        self.act_v = None
        self.pred_v = None

        # Print model info if desired
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
        # Update the n_elements in case it was inferred/adjusted
        self.n_elements = data_loaders.n_elements

        # Grab the DataLoader
        data_loader = data_loaders.get_data_loaders(inference=inference)
        y = data_loader.dataset.y
        if train:
            self.train_len = len(y)
            # Set up a scaler for regression or classification
            if self.classification:
                self.scaler = DummyScaler(y)
            else:
                self.scaler = Scaler(y)
            self.train_loader = data_loader
        self.data_loader = data_loader

    def fit(self, epochs=None, checkin=None, losscurve=False):
    
        # Safety checks
        assert self.train_loader is not None, "Please load training data."
        assert self.data_loader is not None, "Please load validation data."

        # Prepare tracking dictionaries
        self.loss_curve = {'train': [], 'val': []}

        # Set up how frequently we step/cycle LR
        self.epochs_step = 1   # Simplify: step after every epoch
        self.step_size = self.epochs_step * len(self.train_loader)
        print(f'Stepping every {self.step_size} training passes,',
              f'cycling LR every {self.epochs_step} epochs')

        # If not specified, guess a large number of epochs
        if epochs is None:
            n_iterations = 1e4
            epochs = int(n_iterations / len(self.data_loader))
            print(f'Running for {epochs} epochs')

        # If not specified, check in every 2 cycles of self.epochs_step
        if checkin is None:
            checkin = self.epochs_step * 2
            print(f'Checkin at {checkin} epochs to match LR scheduler')

        # Force epochs to align with your cycle length if needed
        # (Below just leaves epochs as-is in this simplified version)
        if epochs % (self.epochs_step * 2) != 0:
            # For example, you could adjust epochs here if you wanted:
            # updated_epochs = epochs - (epochs % (self.epochs_step * 2))
            # epochs = updated_epochs
            pass

        # Define loss function
        self.criterion = RobustL1
        if self.classification:
            print("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss

        # Build an optimizer and SWA wrapper
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)

        # Cyclical LR
        lr_scheduler = CyclicLR(self.optimizer,
                                base_lr=1e-4,
                                max_lr=6e-3,
                                cycle_momentum=False,
                                step_size_up=self.step_size)

        self.swa_start = 2
        self.lr_scheduler = lr_scheduler
        self.stepping = True
        self.lr_list = []
        self.xswa = []
        self.yswa = []
        self.discard_n = 10

        # For early-stopping logic
        self.optimizer.discard_count = 0

        minima = []  # track whether we found "minimum" improvements

        # ---- TRAINING ----
        for epoch in range(epochs):
            self.epoch = epoch
            self.epochs = epochs

            # Switch model to train mode
            self.model.train()
            ti = time()

            for i, data in enumerate(self.train_loader):
                X, y, formula = data
                # Scale targets
                y = self.scaler.scale(y)

                src, frac = X.squeeze(-1).chunk(2, dim=1)

                # Add random noise ("jitter") to fractions for robustness
                frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)
                frac = torch.clamp(frac, 0, 1)
                frac[src == 0] = 0  # zero out "unused" slots
                frac = frac / frac.sum(dim=1, keepdim=True)

                # Move to device
                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)

                # Capture attention if requested at every step
                if self.capture_every == 'step':
                    self.capture_flag = True
                    self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                    self.capture_flag = False

                # Forward pass
                output = self.model.forward(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)

                # Compute loss and backprop
                loss = self.criterion(
                    prediction.view(-1),
                    uncertainty.view(-1),
                    y.view(-1)
                )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # If using cyclical LR
                if self.stepping:
                    self.lr_scheduler.step()

                # SWA checking
                swa_check = (self.epochs_step * self.swa_start - 1)
                epoch_check = ((self.epoch + 1) % (2 * self.epochs_step) == 0)
                learning_time = (epoch_check and self.epoch >= swa_check)
                if learning_time:
                    with torch.no_grad():
                        act_v, pred_v, _, _ = self.predict(self.data_loader)
                    mae_v = mean_absolute_error(act_v, pred_v)
                    self.optimizer.update_swa(mae_v)
                    minima.append(self.optimizer.minimum_found)

            # measure training speed
            dt = time() - ti
            datalen = len(self.train_loader.dataset)
            # print(f'Training speed: {datalen/dt:0.3f} samples/sec')

            # If SWA didn't see improvements, increment discard count
            if learning_time and not any(minima):
                self.optimizer.discard_count += 1
                print(f'Epoch {self.epoch} failed to improve.')
                print(f'Discarded: {self.optimizer.discard_count}/'
                      f'{self.discard_n} weight updates ♻🗑️')
            # Reset minima for next epoch
            minima.clear()

            # ---- END of inline `train()` code ----

            # Still in epoch loop: capture attention every epoch if set
            if self.capture_every == 'epoch':
                self.capture_flag = True
                self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                self.capture_flag = False

            # After each epoch, record and print metrics at checkin points
            self.lr_list.append(self.optimizer.param_groups[0]['lr'])
            if (epoch+1) % checkin == 0 or epoch == epochs - 1 or epoch == 0:
                # Evaluate on train set
                with torch.no_grad():
                    act_t, pred_t, _, _ = self.predict(self.train_loader)
                mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve['train'].append(mae_t)

                # Evaluate on val set
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve['val'].append(mae_v)

                epoch_str = f'Epoch: {epoch}/{epochs} ---'
                train_str = f'train mae: {self.loss_curve["train"][-1]:0.3g}'
                val_str = f'val mae: {self.loss_curve["val"][-1]:0.3g}'

                # If classification, evaluate AUC
                if self.classification:
                    train_auc = roc_auc_score(act_t, pred_t)
                    val_auc = roc_auc_score(act_v, pred_v)
                    train_str = f'train auc: {train_auc:0.3f}'
                    val_str = f'val auc: {val_auc:0.3f}'

                print(epoch_str, train_str, val_str)

                # SWA plotting
                if self.epoch >= (self.epochs_step * self.swa_start - 1):
                    if (self.epoch+1) % (self.epochs_step * 2) == 0:
                        self.xswa.append(self.epoch)
                        self.yswa.append(mae_v)

                # Optionally plot loss curves
                if losscurve:
                    plt.figure(figsize=(8, 5))
                    xval = np.arange(len(self.loss_curve['val'])) * checkin
                    # The first point might be epoch=0, so offset can be 0
                    plt.plot(xval, self.loss_curve['train'],
                             'o-', label='train_mae')
                    plt.plot(xval, self.loss_curve['val'],
                             's--', label='val_mae')
                    plt.plot(self.xswa, self.yswa,
                             'o', ms=12, mfc='none', label='SWA point')
                    plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                    plt.title(f'{self.model_name}')
                    plt.xlabel('epochs')
                    plt.ylabel('MAE')
                    plt.legend()
                    plt.show()

            # Early stopping if needed
            if self.optimizer.discard_count >= self.discard_n:
                print(f'Discarded: {self.optimizer.discard_count}/'
                      f'{self.discard_n} weight updates, '
                      f'early-stopping now 🙅🛑')
                self.optimizer.swap_swa_sgd()
                break

        # If training finished normally, do final SWA swap
        if not (self.optimizer.discard_count >= self.discard_n):
            self.optimizer.swap_swa_sgd()

    def predict(self, loader):
        """
        Runs inference on a given DataLoader. Returns:
        (act, pred, formulae, uncert).
        """
        len_dataset = len(loader.dataset)
        # Each sample is [n_elements, 2]. E.g. if shape = [n_elements, 2].
        n_atoms = int(len(loader.dataset[0][0]) / 2)

        act = np.zeros(len_dataset)
        pred = np.zeros(len_dataset)
        uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=object)

        # For debugging: store raw element indices and fractions
        atoms = np.empty((len_dataset, n_atoms))
        fractions = np.empty((len_dataset, n_atoms))

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula = data

                # If capturing attention, store current formulas
                if self.capture_flag:
                    self.formula_current = None
                    # In some PyTorch versions, formula can be tuple or list
                    if isinstance(formula, tuple):
                        self.formula_current = list(formula)
                    elif isinstance(formula, list):
                        self.formula_current = formula.copy()

                # Split EDM
                src, frac = X.squeeze(-1).chunk(2, dim=1)
                src = src.to(self.compute_device,
                             dtype=torch.long,
                             non_blocking=True)
                frac = frac.to(self.compute_device,
                               dtype=data_type_torch,
                               non_blocking=True)
                y = y.to(self.compute_device,
                         dtype=data_type_torch,
                         non_blocking=True)

                # Forward pass
                output = self.model.forward(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)

                # If using exponent form for uncertainty
                uncertainty = torch.exp(uncertainty) * self.scaler.std

                # Unscale predictions
                prediction = self.scaler.unscale(prediction)

                # If classification, apply sigmoid
                if self.classification:
                    prediction = torch.sigmoid(prediction)

                # Slice location in the arrays
                data_loc = slice(i*self.batch_size,
                                 i*self.batch_size + len(y))

                atoms[data_loc, :] = src.cpu().numpy().astype('int32')
                fractions[data_loc, :] = frac.cpu().numpy().astype('float32')
                act[data_loc] = y.view(-1).cpu().numpy().astype('float32')
                pred[data_loc] = prediction.view(-1).cpu().numpy().astype('float32')
                uncert[data_loc] = uncertainty.view(-1).cpu().numpy().astype('float32')
                formulae[data_loc] = formula

        self.model.train()  # Switch back to train mode (if needed)
        return (act, pred, formulae, uncert)

    def save_network(self, model_name=None):
        """
        Saves the model weights and scaler state
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
        Loads model + scaler state from the .pth file
        """
        path = f'models/trained_models/{path}'
        checkpoint = torch.load(path, map_location=self.compute_device)

        # Recreate the same optimizer structure (SWA, Lookahead, Lamb)
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)

        # Recreate scaler
        self.scaler = Scaler(torch.zeros(3))

        # Load states
        self.model.load_state_dict(checkpoint['weights'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        self.model_name = checkpoint['model_name']
        print(f"Loaded network '{self.model_name}' from {path}")

