import os
import numpy as np
import pandas as pd
import torch
from time import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CyclicLR
from sklearn.metrics import mean_absolute_error, roc_auc_score

from madani.utils.scaling import Scaler, DummyScaler
from madani.utils.optims_update import RobustL1, BCEWithLogitsLoss
from madani.utils.optims_update import Lamb, Lookahead, SWA
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

    def _train_phase(self, phase_name, epochs, checkin, discard_n, optim_params):
        """
        Shared training loop used by both pretraining and fine-tuning phases.
        The `optim_params` dictionary can include parameters such as:
            - base_lr, max_lr: for the LR scheduler
            - swa_start, swa_freq, swa_lr: for SWA settings
            - epochs_step: determines how often to update the scheduler
        """
        print(f"\n--- Starting {phase_name} phase for {epochs} epochs ---")
        # Define loss function (assuming regression by default)
        self.criterion = RobustL1
        if self.classification:
            print("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss

        # Set up optimizer with chosen hyperparameters
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(
            optimizer,
            swa_start=optim_params.get('swa_start', 2),
            swa_freq=optim_params.get('swa_freq', 10),
            swa_lr=optim_params.get('swa_lr', 1e-3)
        )

        # Set LR scheduler parameters; note step_size is computed from the training DataLoader length.
        self.epochs_step = optim_params.get('epochs_step', 1)
        self.step_size = self.epochs_step * len(self.train_loader)
        lr_scheduler = CyclicLR(
            self.optimizer,
            base_lr=optim_params.get('base_lr', 1e-4),
            max_lr=optim_params.get('max_lr', 6e-3),
            cycle_momentum=False,
            step_size_up=self.step_size
        )
        self.lr_scheduler = lr_scheduler
        self.stepping = True
        self.lr_list = []
        self.xswa = []
        self.yswa = []
        self.discard_n = discard_n

        # For early-stopping logic
        self.optimizer.discard_count = 0
        minima = []  # track if a minimum improvement was seen

        # ---- Training loop ----
        for epoch in range(epochs):
            self.epoch = epoch
            self.epochs = epochs
            self.model.train()
            ti = time()

            # Process mini-batches
            for i, data in enumerate(self.train_loader):
                X, y, formula = data
                # Scale targets
                y = self.scaler.scale(y)

                # Split input tensor into source and fraction parts
                src = X[:, :, 0]
                frac = X[:, :, 1]

                # Add random noise ("jitter") to fractions for robustness
                frac = frac * (1 + torch.randn_like(frac) * self.fudge)
                frac = torch.clamp(frac, 0, 1)
                frac[src == 0] = 0  # zero out "unused" slots
                frac = frac / frac.sum(dim=1, keepdim=True)

                # Move data to the compute device
                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)

                # Capture attention if requested on every step
                if self.capture_every == 'step':
                    self.capture_flag = True
                    self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                    self.capture_flag = False

                # Forward pass
                output = self.model.forward(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)

                # Compute loss (reshaping as needed)
                loss = self.criterion(
                    prediction.reshape(-1),
                    uncertainty.reshape(-1),
                    y.reshape(-1)
                )
                loss.backward()
                self.optimizer.step()       # optimizer step (SWA update is managed externally)
                self.optimizer.zero_grad()

                # Step the LR scheduler if using per-step updates
                if self.stepping:
                    self.lr_scheduler.step()

            dt = time() - ti

            # ---- SWA update (after swa_start epochs) ----
            if epoch >= self.optimizer.swa_start:
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.optimizer.update_swa(mae_v)
                minima.append(self.optimizer.minimum_found)

            # Early stopping logic (if no improvement)
            if epoch >= self.optimizer.swa_start and not any(minima):
                self.optimizer.discard_count += 1
                print(f"Epoch {epoch} did not improve. Discard count: {self.optimizer.discard_count}/{discard_n}")
            minima.clear()

            # Capture attention if requested every epoch
            if self.capture_every == 'epoch':
                self.capture_flag = True
                self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                self.capture_flag = False

            self.lr_list.append(self.optimizer.param_groups[0]['lr'])
            if (epoch+1) % checkin == 0 or epoch == epochs - 1 or epoch == 0:
                # Evaluate on training set
                with torch.no_grad():
                    act_t, pred_t, _, _ = self.predict(self.train_loader)
                mae_t = mean_absolute_error(act_t, pred_t)
                # Evaluate on validation set
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                phase_str = f"{phase_name} Epoch {epoch}/{epochs}"
                train_str = f"train mae: {mae_t:0.4g}"
                val_str = f"val mae: {mae_v:0.4g}"
                if self.classification:
                    train_auc = roc_auc_score(act_t, pred_t)
                    val_auc = roc_auc_score(act_v, pred_v)
                    train_str = f"train auc: {train_auc:0.3f}"
                    val_str = f"val auc: {val_auc:0.3f}"
                print(phase_str, train_str, val_str)

                # Track SWA evaluation points for plotting
                if epoch >= self.optimizer.swa_start and (epoch+1) % (self.epochs_step * 2) == 0:
                    self.xswa.append(epoch)
                    self.yswa.append(mae_v)

                # (Optional) Plot loss curves during training
                # Uncomment the following lines if you want to display a plot.
                # plt.figure(figsize=(8, 5))
                # xval = np.arange(len(self.loss_curve['val'])) * checkin
                # plt.plot(xval, self.loss_curve['train'], 'o-', label='train_mae')
                # plt.plot(xval, self.loss_curve['val'], 's--', label='val_mae')
                # plt.plot(self.xswa, self.yswa, 'o', ms=12, mfc='none', label='SWA point')
                # plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                # plt.title(f'{self.model_name} - {phase_name}')
                # plt.xlabel('epochs')
                # plt.ylabel('MAE')
                # plt.legend()
                # plt.show()

            # Early stopping if discard count is reached
            if self.optimizer.discard_count >= discard_n:
                print(f"Discard count reached. Early stopping {phase_name} phase.")
                self.optimizer.swap_swa_sgd()
                break

        # Final SWA swap if early stopping did not trigger
        if self.optimizer.discard_count < discard_n:
            self.optimizer.swap_swa_sgd()

    def pretrain(self, epochs=10, checkin=2, discard_n=10, optim_params=None):
        """
        First phase: Pretraining.
        You can pass a custom dictionary of hyperparameters via optim_params.
        For example, you might use a higher base and max learning rate.
        """
        if optim_params is None:
            optim_params = {
                'base_lr': 1e-3,   # higher learning rate for pretraining
                'max_lr': 1e-2,
                'swa_start': 2,
                'swa_freq': 10,
                'swa_lr': 1e-3,
                'epochs_step': 1,
            }
        print("\n=== Pretraining Phase ===")
        self._train_phase("Pretraining", epochs, checkin, discard_n, optim_params)

    def finetune(self, epochs=20, checkin=2, discard_n=10, optim_params=None):
        """
        Second phase: Fine-tuning.
        Use different hyperparameters here—for instance a lower base LR and a different cyclic scheme.
        The model weights are those resulting from pretraining.
        """
        if optim_params is None:
            optim_params = {
                'base_lr': 1e-4,   # lower learning rate for fine-tuning
                'max_lr': 6e-3,
                'swa_start': 2,
                'swa_freq': 10,
                'swa_lr': 1e-3,
                'epochs_step': 1,
            }
        print("\n=== Fine-tuning Phase ===")
        self._train_phase("Fine-tuning", epochs, checkin, discard_n, optim_params)

    def predict(self, loader_test):
        """
        Runs inference on a given DataLoader. Returns:
        (act, pred, formulae, uncert).
        """
        if isinstance(loader_test, str):
            loader_test = DataHandler(loader_test,
                                     batch_size=2**9,
                                     n_elements=self.n_elements,
                                     inference=True,
                                     verbose=self.verbose,
                                     drop_unary=self.drop_unary,
                                     scale=self.scale)
            loader_test = loader_test.get_data_loaders(inference=True)
        len_dataset = len(loader_test.dataset)

        if len_dataset == 0:
            raise ValueError("The DataLoader is empty. Ensure data is loaded correctly.")
        n_atoms = int(len(loader_test.dataset[0][0]))

        act = np.zeros(len_dataset)
        pred = np.zeros(len_dataset)
        uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=object)

        # For debugging: store raw element indices and fractions
        atoms = np.empty((len_dataset, n_atoms))
        fractions = np.empty((len_dataset, n_atoms))

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader_test):
                X, y, formula = data

                if self.capture_flag:
                    self.formula_current = None
                    if isinstance(formula, tuple):
                        self.formula_current = list(formula)
                    elif isinstance(formula, list):
                        self.formula_current = formula.copy()

                src = X[:, :, 0]
                frac = X[:, :, 1]
                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)

                output = self.model.forward(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)

                # If using exponent form for uncertainty
                uncertainty = torch.exp(uncertainty) * self.scaler.std

                # Unscale predictions
                prediction = self.scaler.unscale(prediction)

                if self.classification:
                    prediction = torch.sigmoid(prediction)

                data_loc = slice(i*self.batch_size,
                                 i*self.batch_size + len(y))
                atoms[data_loc, :] = src.cpu().numpy().astype('int32')
                fractions[data_loc, :] = frac.cpu().numpy().astype('float32')
                act[data_loc] = y.view(-1).cpu().numpy().astype('float32')
                pred[data_loc] = prediction.view(-1).cpu().numpy().astype('float32')
                uncert[data_loc] = uncertainty.view(-1).cpu().numpy().astype('float32')
                formulae[data_loc] = formula

        self.model.train()  # Switch back to training mode if needed
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
        Loads model and scaler state from a .pth file.
        """
        path = f'models/trained_models/{path}'
        checkpoint = torch.load(path, map_location=self.compute_device)

        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)

        self.scaler = Scaler(torch.zeros(3))

        self.model.load_state_dict(checkpoint['weights'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        self.model_name = checkpoint['model_name']
        print(f"Loaded network '{self.model_name}' from {path}")
