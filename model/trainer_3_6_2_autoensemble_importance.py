import os
import copy
import numpy as np
import torch
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, roc_auc_score

from madani.utils.scaling import Scaler, DummyScaler
from madani.utils.optims import RobustL1, BCEWithLogitsLoss
from madani.utils.optims import Lamb, Lookahead
from madani.utils.utils import count_parameters, get_compute_device
from madani.utils.input_processing import DataHandler, data_type_torch
# from madani.model.interpreter

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

        # If the model doesn't specify a device, pick one
        if self.compute_device is None:
            self.compute_device = get_compute_device()

        # Internal flags/vars
        self.capture_flag = False
        self.formula_current = None
        self.act_v = None
        self.pred_v = None

        # We'll store exactly one checkpoint (snapshot) per cycle
        self.snapshots = []

        # For recording checkin points for plotting
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
        Loads data from file_name via DataHandler.
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
            n_cycles=2,            # How many cycles to run
            M=10,                  # Phase I (rapid rise) length
            m=5,                   # Phase II (nose surface) length
            alpha2=1e-4,           # LR offset
            beta1=5e-4,            # slope for rapid rise
            beta2=2e-4,            # slope for nose surface
            checkin=2,             # print/plot interval in epochs
            losscurve=False,
            pretrain_epochs=0):
        """
        Train for multiple cycles. Each cycle has two phases:
          Phase I: M epochs  (lr = alpha2 + beta1 * epoch_in_phase)
          Phase II: m epochs (lr = beta2*((M+m) - epoch_in_cycle) + lr_end_phase1)

        At the end of each cycle, exactly ONE checkpoint is saved to self.snapshots.

        :param n_cycles: number of full cycles to run
        :param M: length of the rapid rise phase
        :param m: length of the nose surface phase
        :param alpha2: base LR offset
        :param beta1: slope for the rapid rise
        :param beta2: slope for the nose surface exploring
        :param checkin: how often (epochs) to print or plot
        :param losscurve: whether to plot the train/val curve
        :param pretrain_epochs: optional number of pretrain epochs at a fixed LR
        """
        assert self.train_loader is not None, "Please load training data (train=True)."
        assert self.data_loader is not None, "Please load validation data (train=False)."

        # Initialize loss trackers
        self.loss_curve = {'train': [], 'val': []}

        # Define the loss function
        self.criterion = RobustL1
        if self.classification:
            print("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss

        # Build optimizer (Lamb + optional Lookahead)
        # base_optim = Lamb(params=self.model.parameters())
        # self.optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = Lamb(params=self.model.parameters())

        # Optional: Pre-training at fixed LR=alpha2
        if pretrain_epochs > 0:
            print(f"Starting pre-training phase for {pretrain_epochs} epoch(s)...")
            for group in self.optimizer.param_groups:
                group['lr'] = alpha2

            for epoch in range(pretrain_epochs):
                self.model.train()
                t0 = time()
                for i, data in enumerate(self.train_loader):
                    X, y, formula = data
                    y = self.scaler.scale(y)

                    src = X[:, :, 0]
                    # print(f'src: {src.shape()}')
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

        # Start multi-cycle training
        total_epochs_per_cycle = M + m
        # We'll track total epochs across all cycles for logging
        self.lr_list = []
        global_epoch = 0  # global epoch counter across cycles

        def set_lr(lr_value):
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr_value

        for cycle_idx in range(n_cycles):
            print(f"\n=== Starting Cycle {cycle_idx+1}/{n_cycles} ===")
            # Precompute LR at end of Phase I for convenience
            lr_end_phase1 = alpha2 + beta1 * (M - 1) if M > 0 else alpha2

            for epoch_in_cycle in range(total_epochs_per_cycle):
                # Determine the current epoch in [0..M+m-1]
                # Phase I: 0..M-1
                # Phase II: M..M+m-1
                if epoch_in_cycle < M:
                    # Rapid rise
                    lr_current = alpha2 + beta1 * epoch_in_cycle
                else:
                    # Nose surface
                    lr_current = beta2 * ((M + m) - epoch_in_cycle) + lr_end_phase1

                lr_current = max(lr_current, 1e-9)  # clamp LR to avoid negatives
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

                global_epoch += 1

                # Checkin for logging/plot
                if (epoch_in_cycle+1) % checkin == 0 or epoch_in_cycle == 0 or epoch_in_cycle == total_epochs_per_cycle-1:
                    with torch.no_grad():
                        act_t, pred_t, _, _ = self.predict(self.train_loader)
                    mae_t = mean_absolute_error(act_t, pred_t)
                    self.loss_curve['train'].append(mae_t)

                    with torch.no_grad():
                        act_v, pred_v, _, _ = self.predict(self.data_loader)
                    mae_v = mean_absolute_error(act_v, pred_v)
                    self.loss_curve['val'].append(mae_v)

                    epoch_str = f'[Cycle {cycle_idx+1} Epoch {epoch_in_cycle}/{total_epochs_per_cycle-1}]'
                    train_str = f'train mae: {mae_t:0.4g}'
                    val_str = f'val mae: {mae_v:0.4g}'
                    if self.classification:
                        train_auc = roc_auc_score(act_t, pred_t)
                        val_auc = roc_auc_score(act_v, pred_v)
                        train_str = f'train auc: {train_auc:0.3f}'
                        val_str = f'val auc: {val_auc:0.3f}'

                    print(f"{epoch_str} {train_str} {val_str}, lr={lr_current:0.6f}")

                    self.xswa.append(global_epoch)
                    self.yswa.append(mae_v)

                    if losscurve:
                        plt.figure(figsize=(8, 5))
                        xval = np.arange(len(self.loss_curve['val'])) * checkin
                        plt.plot(xval, self.loss_curve['train'], 'o-', label='train_mae')
                        plt.plot(xval, self.loss_curve['val'], 's--', label='val_mae')
                        plt.plot(np.array(self.xswa), np.array(self.yswa),
                                 'o', ms=12, mfc='none', label='checkin point')
                        max_val = max(self.loss_curve['val'])
                        plt.ylim(0, 2 * max_val if max_val > 0 else 1)
                        plt.title(f'{self.model_name} (Cycle {cycle_idx+1}/{n_cycles})')
                        plt.xlabel('global steps')
                        plt.ylabel('MAE')
                        plt.legend()
                        plt.show()

            # End of the cycle => Save exactly ONE checkpoint
            snapshot = copy.deepcopy(self.model.state_dict())
            self.snapshots.append(snapshot)
            print(f"Cycle {cycle_idx+1} complete. Saved snapshot #{len(self.snapshots)}")

        print(f"\nAll {n_cycles} cycle(s) finished. Total snapshots: {len(self.snapshots)}")

    def predict(self, loader):
        """
        Runs inference on a given DataLoader.
        Returns: (act, pred, formulae, uncert)
        """
        if isinstance(loader, str):
            loader = DataHandler(loader,
                                 batch_size=2**9,
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
                # print(f'predict:{X[:5]}')
                if self.capture_flag:
                    self.formula_current = list(formula) if isinstance(formula, (tuple, list)) else formula

                src = X[:, :, 0]
                # print(f'src:{src[:5]}')
                frac = X[:, :, 1]
                src = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                y = y.to(self.compute_device, dtype=data_type_torch, non_blocking=True)

                output = self.model(src, frac)
                prediction, uncertainty = output.chunk(2, dim=-1)

                # If using exponent form for uncertainty
                uncertainty = torch.exp(uncertainty) * self.scaler.std
                prediction = self.scaler.unscale(prediction)
                if self.classification:
                    prediction = torch.sigmoid(prediction)

                data_loc = slice(i*self.batch_size, i*self.batch_size + len(y))
                # print(f'data_loc:{data_loc}')
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
        A simple ensemble inference that averages predictions
        across all saved snapshots (one snapshot per cycle).
        """
        if not self.snapshots:
            return self.predict(loader)

        if isinstance(loader, str):
            loader = DataHandler(loader,
                                 batch_size=2**9,
                                 n_elements=self.n_elements,
                                 inference=True,
                                 verbose=self.verbose,
                                 drop_unary=self.drop_unary,
                                 scale=self.scale).get_data_loaders(inference=True)

        len_dataset = len(loader.dataset)
        act_final = None
        preds_list = []
        uncert_list = []
        formulae_final = None

        # Save current state
        current_state = copy.deepcopy(self.model.state_dict())

        for snap in self.snapshots:
            self.model.load_state_dict(snap)
            act_, pred_, formulae_, uncert_ = self.predict(loader)
            preds_list.append(pred_)
            uncert_list.append(uncert_)
            if act_final is None:
                act_final = act_
                formulae_final = formulae_

        # Restore the current model weights
        self.model.load_state_dict(current_state)

        # Average predictions & uncertainties
        ensemble_pred = np.mean(np.stack(preds_list, axis=0), axis=0)
        ensemble_uncert = np.mean(np.stack(uncert_list, axis=0), axis=0)

        return act_final, ensemble_pred, formulae_final, ensemble_uncert

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

        # base_optim = Lamb(params=self.model.parameters())
        # optimizer = Lookahead(base_optimizer=base_optim)
        # self.optimizer = optimizer
        self.optimizer = Lamb(params=self.model.parameters())
        
        self.scaler = Scaler(torch.zeros(3))

        self.model.load_state_dict(checkpoint['weights'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        self.model_name = checkpoint['model_name']
        print(f"Loaded network '{self.model_name}' from {path}")

    def token_knockout_importance(self, src, frac, baseline_token=0):
        """
        Computes token-level importance by knocking out each token (setting it to baseline_token)
        and measuring the change in model output.
        
        Args:
            src (torch.Tensor): Tensor of token indices, shape [B, T].
            frac (torch.Tensor): Fractional tensor, shape [B, T].
            baseline_token (int): The token index used for knockout.
        
        Returns:
            torch.Tensor: Importance scores of shape [B, T].
        """
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(src, frac)  # [B, ...] final output
            B, T = src.shape
            importance_scores = torch.zeros(B, T, device=src.device)
            for b in range(B):
                for t in range(T):
                    src_knock = src.clone()
                    src_knock[b, t] = baseline_token  # Knock out token at position t
                    knockout_output = self.model(src_knock, frac)
                    diff = torch.abs(original_output[b] - knockout_output[b]).mean()
                    importance_scores[b, t] = diff
        return importance_scores

    def feature_knockout_importance(self, src, frac, baseline_option='zero'):
        """
        Computes feature-level importance for the cbfv embedding features by knocking out
        one feature dimension at a time in the embedding matrix. The user can choose the baseline
        value: either 'zero' (using 0.0) or 'mean' (using the column mean).
        
        Args:
            src (torch.Tensor): Tensor of token indices, shape [B, T].
            frac (torch.Tensor): Fractional tensor, shape [B, T].
            baseline_option (str): Either "zero" or "mean". Defaults to "zero".
        
        Returns:
            torch.Tensor: Importance scores for each feature (shape [feat_size]).
        """
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(src, frac)
            # Save the original cbfv weights from the Embedder
            original_weights = self.model.encoder.embed.cbfv.weight.data.clone()  # [vocab_size, feat_size]
            feat_size = original_weights.shape[1]
            feature_importance = torch.zeros(feat_size, device=src.device)
            
            # Pre-compute baseline values for each feature column
            if baseline_option == 'mean':
                baseline_values = original_weights.mean(dim=0)  # [feat_size]
            else:
                baseline_values = torch.zeros(feat_size, device=original_weights.device)
            
            for i in range(feat_size):
                modified_weights = original_weights.clone()
                # Replace column i with the selected baseline value
                modified_weights[:, i] = baseline_values[i]
                # Update the embedding weights temporarily
                self.model.encoder.embed.cbfv.weight.data.copy_(modified_weights)
                knockout_output = self.model(src, frac)
                diff = torch.abs(original_output - knockout_output).mean()
                feature_importance[i] = diff
            
            # Restore the original embedding weights
            self.model.encoder.embed.cbfv.weight.data.copy_(original_weights)
        return feature_importance

    # -----------------------------------------------------------------
    # Internal helper to obtain a DataLoader from CSV path or DataLoader object.
    # -----------------------------------------------------------------
    def _get_loader(self, data_input, batch_size=128):
        if isinstance(data_input, str):
            loader = DataHandler(data_input,
                                 batch_size=batch_size,
                                 n_elements=self.n_elements,
                                 inference=True,
                                 verbose=self.verbose,
                                 drop_unary=self.drop_unary,
                                 scale=self.scale).get_data_loaders(inference=True)
        else:
            loader = data_input
        return loader

    def aggregate_token_knockout_importance(self, data_input, baseline_token=0, batch_size=128):
        """
        Computes token-level knockout importance over the entire dataset (provided as a CSV or DataLoader)
        and aggregates the results by token.
        
        Args:
            data_input (str or DataLoader): CSV file path or DataLoader.
            baseline_token (int): Token used for knockout.
            batch_size (int): Batch size if data_input is a CSV file.
        
        Returns:
            dict: A mapping from token index to its average importance score.
        """
        loader = self._get_loader(data_input, batch_size=batch_size)
        importance_list = []
        src_list = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula = data
                src = X[:, :, 0]  # shape [B, T]
                frac = X[:, :, 1]  # shape [B, T]
                src = src.to(self.compute_device, dtype=torch.long)
                frac = frac.to(self.compute_device, dtype=data_type_torch)
                token_imp = self.token_knockout_importance(src, frac, baseline_token)
                importance_list.append(token_imp.cpu())
                src_list.append(src.cpu())
        
        all_importance = torch.cat(importance_list, dim=0)  # [N, T]
        all_src = torch.cat(src_list, dim=0)                  # [N, T]
        
        token_importance_dict = {}
        token_counts = {}
        B, T = all_importance.shape
        for b in range(B):
            for t in range(T):
                token = int(all_src[b, t].item())
                imp_val = all_importance[b, t].item()
                token_importance_dict[token] = token_importance_dict.get(token, 0.0) + imp_val
                token_counts[token] = token_counts.get(token, 0) + 1
        
        avg_importance = {token: token_importance_dict[token] / token_counts[token] 
                          for token in token_importance_dict}
        return avg_importance

    def token_knockout_importance_data(self, data_input, baseline_token=0, batch_size=128):
        """
        Computes token-level importance for all data in the given DataLoader or CSV file.
        
        Args:
            data_input (str or DataLoader): CSV file path or DataLoader.
            baseline_token (int): Token to use for knockout (default=0).
            batch_size (int): Batch size to use if data_input is a CSV file.
            
        Returns:
            all_importances (torch.Tensor): Concatenated importance scores [N_total, T].
        """
        loader = self._get_loader(data_input, batch_size=batch_size)
        all_importances = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula = data
                src = X[:, :, 0]
                frac = X[:, :, 1]
                src = src.to(self.compute_device, dtype=torch.long)
                frac = frac.to(self.compute_device, dtype=data_type_torch)
                imp = self.token_knockout_importance(src, frac, baseline_token)
                all_importances.append(imp.cpu())
        all_importances = torch.cat(all_importances, dim=0)
        return all_importances

    def feature_knockout_importance_data(self, data_input, baseline_value=0.0, batch_size=128):
        """
        Computes feature-level importance for the cbfv embedding features using all data
        in the given DataLoader or CSV file. The importance is averaged over batches.
        
        Args:
            data_input (str or DataLoader): CSV file path or DataLoader.
            baseline_value (float): Value to replace feature with (default=0.0).
            batch_size (int): Batch size to use if data_input is a CSV file.
            
        Returns:
            avg_importance (torch.Tensor): Tensor of shape [feat_size] with averaged importance scores.
        """
        loader = self._get_loader(data_input, batch_size=batch_size)
        importance_list = []
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula = data
                src = X[:, :, 0]
                frac = X[:, :, 1]
                src = src.to(self.compute_device, dtype=torch.long)
                frac = frac.to(self.compute_device, dtype=data_type_torch)
                imp = self.feature_knockout_importance(src, frac, baseline_value)
                importance_list.append(imp.cpu())
            avg_importance = torch.stack(importance_list, dim=0).mean(dim=0)
        return avg_importance

    
