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
        self.fudge = 0.02  # fraction jitter scaling factor
        self.capture_every = capture_every
        self.verbose = verbose
        self.drop_unary = drop_unary
        self.scale = scale

        if self.compute_device is None:
            self.compute_device = get_compute_device()

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

    def multi_stage_fit(self, stage1_epochs=50, stage2_epochs=100, stage3_epochs=50, checkin=5, losscurve=True, discard_n=10, feature_selection_phase=0.5):
        """
        Multi-stage training:
         - Stage 1: Pretraining without feature selection (selector_scale = 0)
         - Stage 2: Introduce feature selection gradually (e.g. selector_scale = 0.2)
           with added entropy and sparsity penalties.
         - Stage 3: Freeze the concrete layer (using a fixed hard mask) and fine-tune.
           Also prints the selected features and the reduction number.
        """
        # Safety checks
        assert self.train_loader is not None, "Please load training data."
        assert self.data_loader is not None, "Please load validation data."

        total_epochs = stage1_epochs + stage2_epochs + stage3_epochs
        self.loss_curve = {'train': [], 'val': []}

        # Set up LR stepping similar to fit()
        self.epochs_step = 1
        self.step_size = self.epochs_step * len(self.train_loader)
        print(f'Stepping every {self.step_size} training passes')

        print(f'Total epochs: {total_epochs}')
        if checkin is None:
            checkin = self.epochs_step * 2

        # Loss function
        self.criterion = RobustL1
        if self.classification:
            print("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss

        # Build optimizer and SWA wrapper
        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)

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

        # Hyperparameters for penalty terms (you may adjust these)
        lambda_entropy = 0.001
        lambda_sparsity = 0.001

        # -------- Stage 1: Pretraining Without Feature Selection --------
        print("Stage 1: Pretraining without feature selection")
        # Turn off the effect of feature selector by setting selector_scale to 0.
        if hasattr(self.model.encoder.embed, 'selector_scale'):
            self.model.encoder.embed.selector_scale = 0.0

        for epoch in range(stage1_epochs):
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
                output = self.model(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)
                loss = self.criterion(prediction.reshape(-1),
                                      uncertainty.reshape(-1),
                                      y.reshape(-1))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.stepping:
                    self.lr_scheduler.step()
            if (epoch+1) % checkin == 0 or epoch == 0:
                with torch.no_grad():
                    act_t, pred_t, _, _ = self.predict(self.train_loader)
                mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve['train'].append(mae_t)
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve['val'].append(mae_v)
                print(f"Stage 1 - Epoch {epoch}/{stage1_epochs} - train mae: {mae_t:.4g}, val mae: {mae_v:.4g}")

        # -------- Stage 2: Introduce Feature Selection Gradually with Penalties --------
        print("Stage 2: Training with gradual introduction of feature selection and penalties")
        # Re-enable the feature selector and set a modest scaling (e.g., 0.2)
        if hasattr(self.model.encoder.embed, 'selector_scale'):
            self.model.encoder.embed.selector_scale = 0.5

        for epoch in range(stage1_epochs, stage1_epochs+stage2_epochs):
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
                output = self.model(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)
                loss = self.criterion(prediction.reshape(-1),
                                      uncertainty.reshape(-1),
                                      y.reshape(-1))
                # --- Add penalty terms if feature selector exists ---
                if self.model.encoder.embed.feature_selector is not None and self.model.encoder.embed.feature_selector.last_soft_probs is not None:
                    soft_probs = self.model.encoder.embed.feature_selector.last_soft_probs
                    eps = 1e-8
                    entropy = - soft_probs * torch.log(soft_probs + eps) - (1 - soft_probs) * torch.log(1 - soft_probs + eps)
                    entropy_penalty = lambda_entropy * torch.mean(entropy)
                    target_fraction = self.model.encoder.embed.feature_selector.k / self.model.encoder.embed.feat_size
                    sparsity_penalty = lambda_sparsity * (torch.mean(soft_probs) - target_fraction)**2
                    loss = loss + entropy_penalty + sparsity_penalty
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.stepping:
                    self.lr_scheduler.step()
                # Update the temperature of the feature selector.
                if self.model.encoder.embed.feature_selector is not None:
                    self.model.encoder.embed.feature_selector.update_temperature(epoch, total_epochs, feature_selection_phase)
            if ((epoch+1) - stage1_epochs) % checkin == 0 or epoch == stage1_epochs:
                with torch.no_grad():
                    act_t, pred_t, _, _ = self.predict(self.train_loader)
                mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve['train'].append(mae_t)
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve['val'].append(mae_v)
                print(f"Stage 2 - Epoch {epoch}/{stage1_epochs+stage2_epochs} - train mae: {mae_t:.4g}, val mae: {mae_v:.4g}")

        # -------- Stage 3: Fine-Tuning with a Fixed Feature Mask --------
        print("Stage 3: Fine-tuning with fixed feature mask")
        # Extract the stable (hard) mask and freeze it.
        if self.model.encoder.embed.feature_selector is not None:
            hard_mask = self.model.encoder.embed.feature_selector.hard_mask().detach().to(self.compute_device)
            # Print out which features are used and the reduction number.
            selected_indices = (hard_mask > 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
            reduction_number = self.model.encoder.embed.feat_size - len(selected_indices)
            print("Selected feature indices:", selected_indices)
            print(f"Reduction: {reduction_number} features pruned out of {self.model.encoder.embed.feat_size}")
            # Override embedder forward to use the fixed mask.
            def embed_fixed(src):
                x = self.model.encoder.embed.cbfv(src)
                x = x * hard_mask.view(1, 1, -1)
                x_emb = self.model.encoder.embed.fc_mat2vec(x)
                return x_emb
            self.model.encoder.embed.forward = embed_fixed
            # Freeze feature selector parameters.
            self.model.encoder.embed.feature_selector.requires_grad_(False)

        for epoch in range(stage1_epochs+stage2_epochs, total_epochs):
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
                output = self.model(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)
                loss = self.criterion(prediction.reshape(-1),
                                      uncertainty.reshape(-1),
                                      y.reshape(-1))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.stepping:
                    self.lr_scheduler.step()
            if ((epoch+1) - (stage1_epochs+stage2_epochs)) % checkin == 0 or epoch == stage1_epochs+stage2_epochs:
                with torch.no_grad():
                    act_t, pred_t, _, _ = self.predict(self.train_loader)
                mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve['train'].append(mae_t)
                with torch.no_grad():
                    act_v, pred_v, _, _ = self.predict(self.data_loader)
                mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve['val'].append(mae_v)
                print(f"Stage 3 - Epoch {epoch}/{total_epochs} - train mae: {mae_t:.4g}, val mae: {mae_v:.4g}")

        # Finalize training with SWA swap as in the original fit().
        if self.optimizer.discard_count < self.discard_n:
            self.optimizer.swap_swa_sgd()

    def predict(self, loader):
        if isinstance(loader, str):
            from madani.utils.input_processing import DataHandler
            data_handler = DataHandler(loader, batch_size=64, n_elements=self.n_elements,
                                       inference=True, verbose=self.verbose, drop_unary=self.drop_unary, scale=self.scale)
            loader = data_handler.get_data_loaders(inference=True)

        # Temporarily force hard mask during prediction.
        original_embed_forward = self.model.encoder.embed.forward

        def embed_hard(src):
            x = self.model.encoder.embed.cbfv(src)
            if self.model.encoder.embed.feature_selector is not None:
                hard = self.model.encoder.embed.feature_selector.hard_mask().to(self.compute_device)
                hard = hard.view(1, 1, -1)
                x = x * hard
            x_emb = self.model.encoder.embed.fc_mat2vec(x)
            return x_emb

        self.model.encoder.embed.forward = embed_hard
        self.model.eval()
        num_samples = len(loader.dataset)
        sample_X, _, _ = loader.dataset[0]
        n_elements = sample_X.shape[0]
        act = np.zeros(num_samples, dtype=np.float32)
        pred = np.zeros(num_samples, dtype=np.float32)
        uncert = np.zeros(num_samples, dtype=np.float32)
        formulae = np.empty(num_samples, dtype=object)
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
                idx_end = idx_start + batch_size
                pred[idx_start:idx_end] = batch_pred
                act[idx_start:idx_end] = batch_act
                uncert[idx_start:idx_end] = batch_unc
                idx_start = idx_end
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

if __name__ == '__main__':
    # Assume CrabNet is defined elsewhere and uses the above Embedder.
    model = CrabNet(out_dims=3, d_model=512, N=3, heads=4, feature_selector_k=200)
    print(model)
    
    # trainer = ModelTrainer(model, model_name='CrabNet', n_elements=8)
    # # Load your training and validation data
    # trainer.load_data('your_training_data.csv', batch_size=512, train=True)
    # trainer.load_data('your_validation_data.csv', batch_size=512, train=False)
    # # Run the multi-stage training process:
    # trainer.multi_stage_fit(stage1_epochs=50, stage2_epochs=100, stage3_epochs=50, checkin=5, losscurve=True, discard_n=500)
