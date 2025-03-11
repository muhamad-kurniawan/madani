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
	        checkin=2,             # (unused now; loss is recorded every epoch)
	        losscurve=False,
	        pretrain_epochs=0):
	    """
	    Train for multiple cycles with loss and learning rate plotted each epoch.
	    """
	    assert self.train_loader is not None, "Please load training data (train=True)."
	    assert self.data_loader is not None, "Please load validation data (train=False)."
	
	    # Initialize loss trackers and learning rate tracker (per epoch)
	    self.train_loss_all = []
	    self.val_loss_all = []
	    self.lr_list = []
	
	    # Define the loss function
	    self.criterion = RobustL1
	    if self.classification:
	        print("Using BCE loss for classification task")
	        self.criterion = BCEWithLogitsLoss
	
	    # Build optimizer (using Lamb here)
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
	    global_epoch = 0  # Global epoch counter
	
	    def set_lr(lr_value):
	        for pg in self.optimizer.param_groups:
	            pg['lr'] = lr_value
	
	    for cycle_idx in range(n_cycles):
	        print(f"\n=== Starting Cycle {cycle_idx+1}/{n_cycles} ===")
	        lr_end_phase1 = alpha2 + beta1 * (M - 1) if M > 0 else alpha2
	
	        for epoch_in_cycle in range(total_epochs_per_cycle):
	            # Determine the current learning rate
	            if epoch_in_cycle < M:
	                lr_current = alpha2 + beta1 * epoch_in_cycle
	            else:
	                lr_current = beta2 * ((M + m) - epoch_in_cycle) + lr_end_phase1
	            lr_current = max(lr_current, 1e-9)  # clamp LR
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
	
	            # ====== Evaluation pass: record loss at end of epoch ======
	            with torch.no_grad():
	                act_t, pred_t, _, _ = self.predict(self.train_loader)
	                mae_t = mean_absolute_error(act_t, pred_t)
	                act_v, pred_v, _, _ = self.predict(self.data_loader)
	                mae_v = mean_absolute_error(act_v, pred_v)
	            self.train_loss_all.append(mae_t)
	            self.val_loss_all.append(mae_v)
	            global_epoch += 1
	
	            print(f"[Cycle {cycle_idx+1} Epoch {epoch_in_cycle}/{total_epochs_per_cycle-1}] "
	                  f"train mae: {mae_t:0.4g}, val mae: {mae_v:0.4g}, lr: {lr_current:0.6f}")
	
	            # ====== Plot the loss and LR curves (shows all epoch data) ======
	            if losscurve:
	                plt.figure(figsize=(12, 5))
	                # Left: Loss curves
	                plt.subplot(1, 2, 1)
	                epochs = np.arange(len(self.train_loss_all))
	                plt.plot(epochs, self.train_loss_all, marker='o', label='Train MAE')
	                plt.plot(epochs, self.val_loss_all, marker='s', label='Val MAE')
	                plt.title(f'{self.model_name} Loss Curve (Cycle {cycle_idx+1}/{n_cycles})')
	                plt.xlabel('Epoch')
	                plt.ylabel('MAE')
	                plt.legend()
	
	                # Right: Learning Rate curve
	                plt.subplot(1, 2, 2)
	                epochs_lr = np.arange(len(self.lr_list))
	                plt.plot(epochs_lr, self.lr_list, marker='x', label='Learning Rate')
	                plt.title('Learning Rate Schedule')
	                plt.xlabel('Epoch')
	                plt.ylabel('Learning Rate')
	                plt.legend()
	
	                plt.tight_layout()
	                plt.show()
	
	        # End of cycle: Save one snapshot
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
	
    def plot_element_importance_for_formula(
        self,
        data_input,
        formula_index=None,
        formula_str=None,
        baseline_token=0,
        batch_size=128,
        element_symbols=None
    ):
        """
        Gathers data from either a CSV or a DataLoader, computes token-level knockout
        for the entire dataset, and then plots a bar chart of element importance for
        the specified formula. You can specify either a formula_index or a formula_str.

        Args:
            data_input (str or DataLoader): Path to CSV or an existing DataLoader.
            formula_index (int): Which formula (row) to visualize in the aggregated data.
            formula_str (str): The exact formula string to find in the dataset.
            baseline_token (int): Token used for knockout.
            batch_size (int): Batch size if loading from CSV.
            element_symbols (dict): Mapping {token_index: 'ElementSymbol'} for labeling.
        """
        # Ensure user doesn't pass both or neither
        if formula_index is not None and formula_str is not None:
            raise ValueError("Please specify either formula_index or formula_str, not both.")
        
        # Build or retrieve the loader
        loader = self._get_loader(data_input, batch_size=batch_size)

        # Collect all data so we can index formula or search for formula_str
        X_list = []
        formula_list = []
        for i, data in enumerate(loader):
            X_batch, y_batch, f_batch = data
            X_list.append(X_batch)
            formula_list.extend(f_batch)  # store formula strings

        # Concatenate into a single tensor (shape: [N, T, 2])
        X_cat = torch.cat(X_list, dim=0)

        # If formula_str is given, find its index in formula_list
        if formula_str is not None:
            if formula_str in formula_list:
                formula_index = formula_list.index(formula_str)
            else:
                raise ValueError(f"Formula '{formula_str}' not found in dataset.")
        # If neither was provided, default to 0
        if formula_index is None:
            formula_index = 0

        if formula_index >= X_cat.shape[0]:
            raise IndexError(f"formula_index {formula_index} out of range (max {X_cat.shape[0]-1}).")

        # Extract src, frac for the entire dataset
        src_cat = X_cat[:, :, 0].to(self.compute_device, dtype=torch.long)
        frac_cat = X_cat[:, :, 1].to(self.compute_device, dtype=data_type_torch)

        # Compute knockout importance for all formulas
        token_imp_cat = self.token_knockout_importance(src_cat, frac_cat, baseline_token=baseline_token)

        # Select the formula we want to visualize
        src_single = src_cat[formula_index]       # shape: [T]
        imp_single = token_imp_cat[formula_index] # shape: [T]

        # Sort by importance (descending)
        sorted_indices = torch.argsort(imp_single, descending=True)
        sorted_imp = imp_single[sorted_indices]
        sorted_tokens = src_single[sorted_indices]

        # Convert token indices to element labels
        if element_symbols is not None:
            labels = [
                element_symbols.get(int(tok.item()), f"Z{int(tok.item())}") 
                for tok in sorted_tokens
            ]
        else:
            labels = [f"Z{int(tok.item())}" for tok in sorted_tokens]

        # Plot a simple bar chart
        plt.figure(figsize=(8, 4))
        plt.bar(labels, sorted_imp.cpu().numpy(), color='royalblue')
        plt.xlabel("Element")
        plt.ylabel("Knockout Importance")
        
        # If we have formula_list, we can label the plot with the actual formula
        # title_str = f"Element Importance for Formula Index {formula_index}"
        title_str = "Element Importance for Formula"
        if formula_index < len(formula_list):
            title_str += f" {formula_list[formula_index]}"
        plt.title(title_str)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    
