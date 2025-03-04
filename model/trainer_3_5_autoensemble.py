import os
import copy
import numpy as np
import pandas as pd
import torch
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, roc_auc_score

# If these come from madani.* imports, you can keep them. Otherwise adapt:
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

        # If model doesn't specify a device, pick one
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
        """
        Loads data from file_name via DataHandler and sets up self.train_loader
        or self.data_loader accordingly.
        """
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
            if self.classification:
                self.scaler = DummyScaler(y)
            else:
                self.scaler = Scaler(y)
            self.train_loader = data_loader
        else:
            self.data_loader = data_loader

    def fit(self,
            M=10,                     # Phase I (rapid rise) length
            m=5,                      # Phase II (nose surface) length
            alpha2=1e-4,             # base LR offset
            beta1=5e-4,              # slope for rapid-rise
            beta2=2e-4,              # slope for nose-surface
            checkin=2,               # how often (in epochs) to evaluate & maybe capture snapshot
            losscurve=False,
            adaptive_threshold=0.01,
            pretrain_epochs=0):
        """
        Trains the model in two phases using the "rapid rise" schedule:

          Phase I (epochs = 0..M-1):
             lr(n) = alpha2 + beta1 * n
          Phase II (epochs = M..M+m-1):
             lr(n) = beta2 * ((M+m) - n) + [alpha2 + beta1*(M-1)]

        If pretrain_epochs > 0, we run a separate pre-training loop (with a
        simple LR or a small cycLR) before these two phases.

        If the relative improvement over the best validation MAE is < adaptive_threshold
        at checkin, we capture a snapshot for the ensemble.
        """
        assert self.train_loader is not None, "Please load training data (train=True)."
        assert self.data_loader is not None, "Please load validation data (train=False)."

        # Initialize the loss curve trackers
        self.loss_curve = {'train': [], 'val': []}

        # Define the loss function
        self.criterion = RobustL1
        if self.classification:
            print("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss

        # Build optimizer with Lamb + Lookahead
        # base_optim = Lamb(params=self.model.parameters())
        # self.optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = Lamb(params=self.model.parameters())

        # Optional: pre-training phase
        if pretrain_epochs > 0:
            print(f"Starting pre-training phase for {pretrain_epochs} epoch(s)...")
            # For simplicity, we'll just fix LR=alpha2 or do a mini cyc. Let's fix it:
            for group in self.optimizer.param_groups:
                group['lr'] = alpha2

            for epoch in range(pretrain_epochs):
                self.model.train()
                t0 = time()
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

                    self.optimizer.zero_grad()
                    output = self.model(src, frac)
                    prediction, uncertainty = output.chunk(2, dim=-1)
                    loss = self.criterion(
                        prediction.view(-1),
                        uncertainty.view(-1),
                        y.view(-1)
                    )
                    loss.backward()
                    self.optimizer.step()

                dt = time() - t0
                print(f"[Pretrain] Epoch {epoch+1}/{pretrain_epochs}, took {dt:.2f}s")
            print("Pre-training phase completed.\n")

        # Start main 2-phase training
        total_epochs = M + m
        self.best_val = float('inf')
        self.lr_list = []

        # Helper function to set LR
        def set_lr(lr_value):
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr_value

        # Precompute lr_now at M-1 for convenience
        # (lr at end of Phase I)
        lr_end_phase1 = alpha2 + beta1 * (M - 1) if M > 0 else alpha2

        for epoch in range(total_epochs):
            # Compute LR for this epoch
            if epoch < M:
                # Phase I
                lr_current = alpha2 + beta1 * epoch
            else:
                # Phase II
                lr_current = beta2 * ((M + m) - epoch) + lr_end_phase1

            # Ensure LR doesn't go negative if parameters are weird
            lr_current = max(lr_current, 1e-9)
            set_lr(lr_current)
            self.lr_list.append(lr_current)

            # ====== Training pass ======
            self.model.train()
            t0 = time()
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

                self.optimizer.zero_grad()
                output = self.model(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)
                loss = self.criterion(
                    prediction.reshape(-1),
                    uncertainty.reshape(-1),
                    y.reshape(-1)
                )
                loss.backward()
                self.optimizer.step()

            dt = time() - t0

            if self.capture_every == 'epoch':
                self.capture_flag = True
                self.act_v, self.pred_v, _, _ = self.predict(self.data_loader)
                self.capture_flag = False

            # ====== Checkin & snapshot logic ======
            if (epoch + 1) % checkin == 0 or epoch == total_epochs - 1 or epoch == 0:
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

                epoch_str = f'Epoch: {epoch}/{total_epochs-1} ---'
                train_str = f'train mae: {mae_t:0.4g}'
                val_str = f'val mae: {mae_v:0.4g}'
                if self.classification:
                    train_auc = roc_auc_score(act_t, pred_t)
                    val_auc = roc_auc_score(act_v, pred_v)
                    train_str = f'train auc: {train_auc:0.3f}'
                    val_str = f'val auc: {val_auc:0.3f}'

                print(f"{epoch_str} {train_str} {val_str}, lr={lr_current:0.6f}")

                self.xswa.append(epoch)
                self.yswa.append(mae_v)

                # Adaptive snapshot capture
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

                # Optionally plot loss curves
                if losscurve:
                    plt.figure(figsize=(8, 5))
                    xval = np.arange(len(self.loss_curve['val'])) * checkin
                    plt.plot(xval, self.loss_curve['train'], 'o-', label='train_mae')
                    plt.plot(xval, self.loss_curve['val'], 's--', label='val_mae')
                    plt.plot(np.array(self.xswa), np.array(self.yswa),
                             'o', ms=12, mfc='none', label='checkin point')
                    # Keep the y-limit somewhat reasonable
                    max_val = max(self.loss_curve['val'])
                    plt.ylim(0, 2 * max_val if max_val > 0 else 1)
                    plt.title(f'{self.model_name}')
                    plt.xlabel('epochs')
                    plt.ylabel('MAE')
                    plt.legend()
                    plt.show()

        print(f"Training finished. Captured {len(self.snapshots)} snapshot(s).")

    def predict(self, loader):
        """
        Runs inference on a given DataLoader.
        Returns: (act, pred, formulae, uncert)
        """
        # If the user passes a string, build a DataLoader
        if isinstance(loader, str):
            loader = DataHandler(loader,
                                 batch_size=64,
                                 n_elements=self.n_elements,
                                 inference=True,
                                 verbose=self.verbose,
                                 drop_unary=self.drop_unary,
                                 scale=self.scale).get_data_loaders(inference=True)
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

                output = self.model(src, frac)
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

    def ensemble_predict(self, loader):
        """
        Perform an ensemble prediction by averaging predictions from all snapshots.
        If no snapshots, just do a single forward pass with the current model.
        """
        if not self.snapshots:
            return self.predict(loader)

        # Convert loader if it's a string
        if isinstance(loader, str):
            loader = DataHandler(loader,
                                 batch_size=64,
                                 n_elements=self.n_elements,
                                 inference=True,
                                 verbose=self.verbose,
                                 drop_unary=self.drop_unary,
                                 scale=self.scale).get_data_loaders(inference=True)

        len_dataset = len(loader.dataset)
        n_atoms = int(len(loader.dataset[0][0]))

        act = np.zeros(len_dataset)
        ensemble_pred = np.zeros(len_dataset)
        ensemble_uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=object)

        # Save current model state
        current_state = copy.deepcopy(self.model.state_dict())

        # We'll accumulate predictions from each snapshot
        snapshot_preds = []
        snapshot_uncerts = []

        for snap in self.snapshots:
            self.model.load_state_dict(snap)
            act_, pred_, formulae_, uncert_ = self.predict(loader)
            snapshot_preds.append(pred_)
            snapshot_uncerts.append(uncert_)
            # We'll just use the first snapshot's 'act' & 'formulae' as reference
            # because they should all match in ordering
            act = act_
            formulae = formulae_

        # Restore the model's current state
        self.model.load_state_dict(current_state)

        # Average predictions & uncertainties
        ensemble_pred = np.mean(np.stack(snapshot_preds, axis=0), axis=0)
        ensemble_uncert = np.mean(np.stack(snapshot_uncerts, axis=0), axis=0)

        return act, ensemble_pred, formulae, ensemble_uncert

    def save_network(self, model_name=None):
        """
        Saves the current model state and scaler state to disk.
        Snapshots are not automatically saved here, but you can add them if you want.
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
        Loads model weights and scaler state from the .pth file
        (not snapshots).
        """
        path = f'models/trained_models/{path}'
        checkpoint = torch.load(path, map_location=self.compute_device)

        # base_optim = Lamb(params=self.model.parameters())
        # optimizer = Lookahead(base_optimizer=base_optim)
        # self.optimizer = optimizer
        self.optimizer = Lamb(params=self.model.parameters())

        self.scaler = Scaler(torch.zeros(3))

        self.model.load_state_dict(checkpoint['weights'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        self.model_name = checkpoint['model_name']
        print(f"Loaded network '{self.model_name}' from {path}")
