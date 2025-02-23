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

    def fit(self, epochs=None, checkin=None, losscurve=False, discard_n=10, feature_selection_phase=0.5):
    
        # Safety checks
        assert self.train_loader is not None, "Please load training data."
        assert self.data_loader is not None, "Please load validation data."

        # Prepare tracking dictionaries
        self.loss_curve = {'train': [], 'val': []}

        # Set up how frequently we step/cycle LR
        self.epochs_step = 1   # step after every epoch
        self.step_size = self.epochs_step * len(self.train_loader)
        print(f'Stepping every {self.step_size} training passes, cycling LR every {self.epochs_step} epochs')

        # If epochs not specified, guess a large number
        if epochs is None:
            n_iterations = 1e4
            epochs = int(n_iterations / len(self.data_loader))
            print(f'Running for {epochs} epochs')

        # If checkin not specified, set default
        if checkin is None:
            checkin = self.epochs_step * 2
            print(f'Checkin at {checkin} epochs to match LR scheduler')

        # Loss function
        self.criterion = RobustL1
        if self.classification:
            print("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss

        # Build optimizer and SWA wrapper
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)

        # Cyclical LR scheduler
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
        self.discard_n = discard_n

        self.optimizer.discard_count = 0
        minima = []  # track improvements

        # ---- TRAINING LOOP ----
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
                frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)
                frac = torch.clamp(frac, 0, 1)
                frac[src == 0] = 0
                frac = frac / frac.sum(dim=1, keepdim=True)

                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)

                output = self.model(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)

                loss = self.criterion(prediction.reshape(-1),
                                      uncertainty.reshape(-1),
                                      y.reshape(-1))
                lambda_entropy = 0.01
                penalty = lambda_entropy * self.model.encoder.embed.feature_selector.entropy_penalty

                loss = loss + penalty
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.stepping:
                    self.lr_scheduler.step()

                swa_check = (self.epochs_step * self.swa_start - 1)
                epoch_check = ((self.epoch + 1) % (2 * self.epochs_step) == 0)
                learning_time = (epoch_check and self.epoch >= swa_check)
                if learning_time:
                    with torch.no_grad():
                        act_v, pred_v, _, _ = self.predict(self.data_loader)
                    mae_v = mean_absolute_error(act_v, pred_v)
                    self.optimizer.update_swa(mae_v)
                    minima.append(self.optimizer.minimum_found)
            dt = time() - ti

            if learning_time and not any(minima):
                self.optimizer.discard_count += 1
                print(f'Epoch {self.epoch} failed to improve.')
                print(f'Discarded: {self.optimizer.discard_count}/{self.discard_n} weight updates.')
            minima.clear()

            # Update the temperature of feature selector
            if self.model.encoder.embed.feature_selector is not None:
                self.model.encoder.embed.feature_selector.update_temperature(epoch, epochs, feature_selection_phase)

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
                train_str = f'train mae: {self.loss_curve["train"][-1]:0.4g}'
                val_str = f'val mae: {self.loss_curve["val"][-1]:0.4g}'
                if self.classification:
                    train_auc = roc_auc_score(act_t, pred_t)
                    val_auc = roc_auc_score(act_v, pred_v)
                    train_str = f'train auc: {train_auc:0.3f}'
                    val_str = f'val auc: {val_auc:0.3f}'
                print(epoch_str, train_str, val_str)

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

                    # At checkin time, print the hard mask from the feature selector.
                    if hasattr(self.model.encoder.embed, 'feature_selector'):
                        print(f'hard mask: {self.model.encoder.embed.feature_selector.hard_mask()}')
            if self.optimizer.discard_count >= self.discard_n:
                print(f'Early stopping: Discarded {self.optimizer.discard_count}/{self.discard_n} weight updates.')
                self.optimizer.swap_swa_sgd()
                break

        if not (self.optimizer.discard_count >= self.discard_n):
            self.optimizer.swap_swa_sgd()

    def predict(self, loader):
        """
        Runs inference on a given DataLoader.
        Returns:
            (act, pred, formulae, uncert)
        """
        if isinstance(loader, str):
            from madani.utils.input_processing import DataHandler
            data_handler = DataHandler(loader, batch_size=64, n_elements=self.n_elements,
                                       inference=True, verbose=self.verbose, drop_unary=self.drop_unary, scale=self.scale)
            loader = data_handler.get_data_loaders(inference=True)

        # --- Force hard mask during prediction ---
        # Save the original embedder forward method.
        original_embed_forward = self.model.encoder.embed.forward

        # Define a new forward method that applies the hard mask.
        def embed_hard(src):
            x = self.model.encoder.embed.cbfv(src)
            if self.model.encoder.embed.feature_selector is not None:
                # Get the hard mask (shape: [feat_size]) and expand it.
                hard = self.model.encoder.embed.feature_selector.hard_mask().to(self.compute_device)
                hard = hard.view(1, 1, -1)
                x = x * hard
            x_emb = self.model.encoder.embed.fc_mat2vec(x)
            return x_emb

        # Replace the forward method temporarily.
        self.model.encoder.embed.forward = embed_hard

        self.model.eval()
        num_samples = len(loader.dataset)
        sample_X, _, _ = loader.dataset[0]
        n_elements = sample_X.shape[0]

        act = np.zeros(num_samples, dtype=np.float32)
        pred = np.zeros(num_samples, dtype=np.float32)
        uncert = np.zeros(num_samples, dtype=np.float32)
        formulae = np.empty(num_samples, dtype=object)

        atoms = np.zeros((num_samples, n_elements), dtype=np.int32)
        fractions = np.zeros((num_samples, n_elements), dtype=np.float32)

        idx_start = 0
        with torch.no_grad():
            for (X, y, formula) in loader:
                batch_size = X.shape[0]
                src = X[:, :, 0].to(self.compute_device, dtype=torch.long)
                frac = X[:, :, 1].to(self.compute_device, dtype=data_type_torch)
                y = y.to(self.compute_device, dtype=data_type_torch)

                frac[src == 0] = 0
                frac_sum = frac.sum(dim=1, keepdim=True)
                frac_sum[frac_sum == 0] = 1e-9
                frac = frac / frac_sum

                output = self.model(src, frac)
                prediction, log_uncert = output.chunk(2, dim=-1)

                if self.classification:
                    prediction = torch.sigmoid(prediction)
                    uncertainty_ = torch.sigmoid(log_uncert)
                else:
                    uncertainty_ = torch.exp(log_uncert) * self.scaler.std
                    prediction = self.scaler.unscale(prediction)

                batch_pred = prediction.view(-1).cpu().numpy().astype(np.float32)
                batch_act = y.view(-1).cpu().numpy().astype(np.float32)
                batch_unc = uncertainty_.view(-1).cpu().numpy().astype(np.float32)
                batch_src = src.cpu().numpy().astype(np.int32)
                batch_frac = frac.cpu().numpy().astype(np.float32)

                idx_end = idx_start + batch_size
                pred[idx_start:idx_end] = batch_pred
                act[idx_start:idx_end] = batch_act
                uncert[idx_start:idx_end] = batch_unc
                atoms[idx_start:idx_end, :] = batch_src
                fractions[idx_start:idx_end, :] = batch_frac

                if isinstance(formula, (list, tuple)):
                    formulae[idx_start:idx_end] = formula
                else:
                    formulae[idx_start:idx_end] = [formula] * batch_size
                idx_start = idx_end

        # Restore the original embedder forward method.
        self.model.encoder.embed.forward = original_embed_forward
        self.model.train()
        return act, pred, formulae, uncert

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
        from madani.utils.optims import Lamb, Lookahead, SWA
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)
        self.scaler = Scaler(torch.zeros(3))
        self.model.load_state_dict(checkpoint['weights'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        self.model_name = checkpoint['model_name']
        print(f"Loaded network '{self.model_name}' from {path}")

# Example usage
if __name__ == '__main__':
    # For instance, assume d_model=512 for the transformer,
    # and we wish to select 200 features from the raw element vector.
    model = CrabNet(out_dims=3, d_model=512, N=3, heads=4, feature_selector_k=200)
    print(model)
    
    trainer = ModelTrainer(model, model_name='CrabNet', n_elements=8)
    # Load your training and validation data using trainer.load_data(...)
    # Here we assume that the data loader is set up appropriately.
    # Then call:
    trainer.fit(epochs=1000, checkin=5, losscurve=True, discard_n=500)
