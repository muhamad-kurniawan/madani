import os
import copy
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from madani.utils.scaling import Scaler, DummyScaler
from madani.utils.optims_update import RobustL1, BCEWithLogitsLoss, Lamb
from madani.utils.utils import count_parameters, get_compute_device
from madani.utils.input_processing import DataHandler, data_type_torch


class ModelTrainer:
    """Refactored trainer with **one global optimiser config** that every phase
    uses by default.  *variant* methods no longer *require* a list; if
    `variant_configs` is omitted or empty they will simply fine‑tune **one**
    variant that also uses the default parameters.
    """

    # ───────────────────────────────────────────────────────
    #  INIT
    # ───────────────────────────────────────────────────────
    def __init__(self,
                 model,
                 model_name: str = "UnnamedModel",
                 n_elements: int = 8,
                 capture_every=None,
                 verbose: bool = True,
                 drop_unary: bool = True,
                 scale: bool = True,
                 default_optim_params: dict | None = None):

        # Store user arguments
        self.model = model
        self.model_name = model_name
        self.n_elements = n_elements
        self.capture_every = capture_every
        self.verbose = verbose
        self.drop_unary = drop_unary
        self.scale = scale
        self.classification = False

        # Global default optimiser parameters (shared by *every* phase)
        self.default_optim_params = default_optim_params or dict(
            base_lr=1e-4, max_lr=6e-3, lr_decay=1.0, epochs_step=1
        )

        # Misc helpers
        self.fudge = 0.02  # fraction “jitter” noise strength
        self.compute_device = getattr(model, "compute_device", None) or get_compute_device()
        self.capture_flag = False

        if verbose:
            print("\nModel architecture: out_dims, d_model, N, heads")
            print(f"{model.out_dims}, {model.d_model}, {model.N}, {model.heads}")
            print(f"Running on compute device: {self.compute_device}")
            print(f"Model size: {count_parameters(model)} parameters\n")
            if capture_every is not None:
                print(f"Capturing attention tensors every {capture_every}")

    # ───────────────────────────────────────────────────────
    #  HELPER: merge defaults with overrides
    # ───────────────────────────────────────────────────────
    def _resolve_optim_params(self, override: dict | None) -> dict:
        """Return the union of default optimiser params and a per‑call override."""
        override = override or {}
        return {**self.default_optim_params, **override}

    def set_default_optim_params(self, **kwargs):
        """Update optimiser defaults in‑place."""
        self.default_optim_params.update(kwargs)
        if self.verbose:
            print("[ModelTrainer] New default optimiser params:", self.default_optim_params)

    # ───────────────────────────────────────────────────────
    #  DATA LOADING (unchanged except typing tweaks)
    # ───────────────────────────────────────────────────────
    def load_data(self, file_name: str, *, batch_size: int = 2 ** 9, train: bool = False):
        self.batch_size = batch_size
        loader = DataHandler(
            file_name,
            batch_size=batch_size,
            n_elements=self.n_elements,
            inference=not train,
            verbose=self.verbose,
            drop_unary=self.drop_unary,
            scale=self.scale,
        )
        if self.verbose:
            print(f"Loading data with up to {loader.n_elements:0.0f} elements in the formula")
        self.n_elements = loader.n_elements
        self.data_loader = loader.get_data_loaders(inference=not train)
        if train:
            y = self.data_loader.dataset.y
            self.train_loader = self.data_loader
            self.scaler = DummyScaler(y) if self.classification else Scaler(y)

    # ───────────────────────────────────────────────────────
    #  CORE TRAINING LOOP
    # ───────────────────────────────────────────────────────
    def _train_phase(self, phase_name: str, *, epochs: int, checkin: int, optim_params: dict | None):
        optim_params = self._resolve_optim_params(optim_params)
        print(f"\n--- {phase_name} for {epochs} epochs — optim params: {optim_params}")

        self.criterion = RobustL1 if not self.classification else BCEWithLogitsLoss
        self.optimizer = Lamb(params=self.model.parameters())

        # Build scheduler
        base_lr, max_lr = optim_params["base_lr"], optim_params["max_lr"]
        lr_decay = optim_params["lr_decay"]
        epochs_step = optim_params["epochs_step"]
        step_size = epochs_step * len(self.train_loader)
        make_sched = lambda lo, hi: torch.optim.lr_scheduler.CyclicLR(
            self.optimizer, base_lr=lo, max_lr=hi, step_size_up=step_size, cycle_momentum=False
        )
        self.lr_scheduler = make_sched(base_lr, max_lr)

        best_score, best_state = float("inf"), copy.deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            self.model.train()
            for X, y, _ in self.train_loader:
                # jitter fractions
                y = self.scaler.scale(y)
                src, frac = X[:, :, 0], X[:, :, 1]
                frac = torch.clamp(frac * (1 + torch.randn_like(frac) * self.fudge), 0, 1)
                frac[src == 0] = 0
                frac = frac / frac.sum(dim=1, keepdim=True)
                src = src.to(self.compute_device, dtype=torch.long)
                frac = frac.to(self.compute_device, dtype=data_type_torch)
                y = y.to(self.compute_device, dtype=data_type_torch)

                pred, log_u = self.model(src, frac).chunk(2, dim=-1)
                loss = self.criterion(pred.view(-1), log_u.view(-1), y.view(-1))
                loss.backward()
                self.optimizer.step(); self.optimizer.zero_grad(); self.lr_scheduler.step()

            if (epoch + 1) % checkin == 0 or epoch in (0, epochs - 1):
                mae = mean_absolute_error(*self.predict(self.data_loader)[:2])
                print(f"    Epoch {epoch:03d} — val MAE={mae:.4g}")
                if mae < best_score:
                    best_score, best_state = mae, copy.deepcopy(self.model.state_dict())
                    print("      ↳ new best")

            if lr_decay < 1.0:
                base_lr *= lr_decay; max_lr *= lr_decay
                self.lr_scheduler = make_sched(base_lr, max_lr)

        self.model.load_state_dict(best_state)
        print(f"Loaded best {phase_name} model (MAE={best_score:.4g})")

    # ───────────────────────────────────────────────────────
    #  PUBLIC WRAPPERS
    # ───────────────────────────────────────────────────────
    def pretrain(self, *, epochs: int = 10, checkin: int = 2, optim_params: dict | None = None):
        self._train_phase("Pretrain", epochs=epochs, checkin=checkin, optim_params=optim_params)

    def finetune(self, *, epochs: int = 20, checkin: int = 2, optim_params: dict | None = None):
        self._train_phase("Finetune", epochs=epochs, checkin=checkin, optim_params=optim_params)

    # ───────────────────────────────────────────────────────
    #  SOUP HELPERS — variant_configs optional
    # ───────────────────────────────────────────────────────
    def _run_variant(self, override, *, epochs, checkin):
        merged = self._resolve_optim_params(override)
        self.finetune(epochs=epochs, checkin=checkin, optim_params=merged)
        mae = mean_absolute_error(*self.predict(self.data_loader)[:2])
        return copy.deepcopy(self.model.state_dict()), mae

    def finetune_variants_for_soup(self, variant_configs: list | None = None, *, epochs: int = 20, checkin: int = 2):
        """If *variant_configs* is None or empty, run **one** variant using defaults."""
        variant_configs = variant_configs or [None]
        pretrained = copy.deepcopy(self.model.state_dict())
        states = []
        for i, cfg in enumerate(variant_configs):
            print(f"\n--- Variant {i+1}/{len(variant_configs)} ---")
            self.model.load_state_dict(pretrained)
            st, _ = self._run_variant(cfg, epochs=epochs, checkin=checkin)
            states.append(st)
        avg = {k: sum(st[k] for st in states) / len(states) for k in states[0]}
        self.model.load_state_dict(avg)
        print(f"\n[Plain Soup] Averaged {len(states)} variant(s) — loaded into model")

    def finetune_variants_for_greedy_soup(self, variant_configs, epochs=20, checkin=2):
        """
        Starting from the pretrained model, perform several fine-tuning runs using variant
        hyperparameters. Then, construct a greedy model soup: start with the best candidate,
        and incrementally add models if their inclusion (averaged with the current soup) reduces 
        the validation error (here, mean absolute error).
        
        Args:
            variant_configs (list of dict): List of hyperparameter dictionaries.
        """
        print("\n=== Finetuning Variants for Greedy Soup ===")
        # Save the pretrained state
        pretrained_state = self.model.state_dict()
        candidate_state_dicts = []
        candidate_scores = []

        # Fine-tune each variant and evaluate on the validation set.
        for i, config in enumerate(variant_configs):
            print(f"\n--- Variant {i+1}/{len(variant_configs)} ---")
            self.model.load_state_dict(pretrained_state)
            self.finetune(epochs=epochs, checkin=checkin, optim_params=config)
            # Get predictions on validation set and compute error
            act_v, pred_v, _, _ = self.predict(self.data_loader)
            mae_v = mean_absolute_error(act_v, pred_v)
            # candidate_state_dicts.append(self.model.state_dict().copy())
            candidate_state_dicts.append(copy.deepcopy(self.model.state_dict()))
            candidate_scores.append(mae_v)
            print(f"Variant {i+1} validation MAE: {mae_v}")

        # Begin greedy soup: start with the best candidate (lowest MAE)
        best_index = np.argmin(candidate_scores)
        print(f"\nStarting greedy soup with candidate {best_index+1} (MAE: {candidate_scores[best_index]})")
        current_soup = candidate_state_dicts[best_index]
        current_count = 1
        current_score = candidate_scores[best_index]
        greedy_indices = [best_index]

        # Sort candidate indices by increasing MAE (best to worst)
        sorted_indices = sorted(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
        for idx in sorted_indices:
            if idx == best_index:
                continue  # already in the soup
            # Construct a new soup by averaging current soup with the candidate's weights
            new_soup = {}
            for key in current_soup.keys():
                new_soup[key] = (current_soup[key] * current_count + candidate_state_dicts[idx][key]) / (current_count + 1)
            # Evaluate the new soup on validation data
            self.model.load_state_dict(new_soup)
            act_v, pred_v, _, _ = self.predict(self.data_loader)
            new_score = mean_absolute_error(act_v, pred_v)
            print(f"Trying candidate {idx+1}: new MAE = {new_score} vs current MAE = {current_score}")
            if new_score < current_score:
                # Accept the candidate into the soup
                print(f"Candidate {idx+1} accepted.")
                current_soup = new_soup
                current_count += 1
                current_score = new_score
                greedy_indices.append(idx)
            else:
                print(f"Candidate {idx+1} rejected.")

        # Load the final greedy soup into the model
        self.model.load_state_dict(current_soup)
        print("\nGreedy model soup has been created and loaded into the model.")
        print("Included candidate models:", [i+1 for i in greedy_indices])

    def finetune_variants_for_random_soup(self, variant_configs, epochs=20, checkin=2, num_samples=10):
        """
        Starting from the pretrained model, perform several fine-tuning runs using variant
        hyperparameters. Then, generate a number of random combinations of the candidate 
        models' weights and choose the soup that yields the lowest validation MAE.
        
        Args:
            variant_configs (list of dict): List of hyperparameter dictionaries.
            num_samples (int): Number of random weight combinations to try.
        """
        print("\n=== Finetuning Variants for Random Soup ===")
        # Save the pretrained state
        pretrained_state = self.model.state_dict()
        candidate_state_dicts = []
        candidate_scores = []

        # Fine-tune each variant and evaluate on the validation set.
        for i, config in enumerate(variant_configs):
            print(f"\n--- Variant {i+1}/{len(variant_configs)} ---")
            self.model.load_state_dict(pretrained_state)
            self.finetune(epochs=epochs, checkin=checkin, optim_params=config)
            act_v, pred_v, _, _ = self.predict(self.data_loader)
            
            mae_v = mean_absolute_error(act_v, pred_v)
            # candidate_state_dicts.append(self.model.state_dict().copy())
            candidate_state_dicts.append(copy.deepcopy(self.model.state_dict()))
            candidate_scores.append(mae_v)
            print(f"Variant {i+1} validation MAE: {mae_v}")

        n_candidates = len(candidate_state_dicts)
        best_random_score = float('inf')
        best_random_soup = None
        best_coeffs = None

        # Try a number of random combinations
        for sample in range(num_samples):
            # Create random coefficients that sum to 1
            coeffs = np.random.dirichlet(np.ones(n_candidates))
            random_soup = {}
            for key in candidate_state_dicts[0].keys():
                # Compute the weighted average for this parameter across all candidate models
                weighted_sum = sum(coeffs[i] * candidate_state_dicts[i][key] for i in range(n_candidates))
                random_soup[key] = weighted_sum
            # Load the random soup and evaluate on the validation set
            self.model.load_state_dict(random_soup)
            act_v, pred_v, _, _ = self.predict(self.data_loader)

            random_mae = mean_absolute_error(act_v, pred_v)
            print(f"Random soup sample {sample+1}: MAE = {random_mae}")
            if random_mae < best_random_score:
                best_random_score = random_mae
                best_random_soup = random_soup
                best_coeffs = coeffs

        # Load the best-performing random soup into the model
        self.model.load_state_dict(best_random_soup)
        print(f"\nBest random soup achieved MAE: {best_random_score}")
        print("Best soup coefficients (per candidate):", best_coeffs)
         
    def predict(self, loader_test):
        """
        Runs inference on a given DataLoader.
        Returns: (actual, prediction, formulae, uncertainty).
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
        atoms = np.empty((len_dataset, n_atoms))
        fractions = np.empty((len_dataset, n_atoms))

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader_test):
                X, y, formula = data
                if self.capture_flag:
                    self.formula_current = list(formula) if isinstance(formula, (tuple, list)) else formula
                src = X[:, :, 0].to(self.compute_device, dtype=torch.long, non_blocking=True)
                frac = X[:, :, 1].to(self.compute_device, dtype=data_type_torch, non_blocking=True)
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
        self.model.train()
        return (act, pred, formulae, uncert)

    def save_network(self, model_name=None):
        """
        Saves the model weights and scaler state.
        """
        if model_name is None:
            model_name = self.model_name
        os.makedirs('models/trained_models', exist_ok=True)
        path = f'models/trained_models/{model_name}.pth'
        print(f"Saving network ({model_name}) to {path}")
        save_dict = {
            'weights': self.model.state_dict(),
            'scaler_state': self.scaler.state_dict(),
            'model_name': model_name
        }
        torch.save(save_dict, path)

    def load_network(self, path):
        """
        Loads model and scaler state from a checkpoint file.
        """
        path = os.path.join('models', 'trained_models', path)
        checkpoint = torch.load(path, map_location=self.compute_device)
        base_optim = Lamb(params=self.model.parameters())
        # optimizer = Lookahead(base_optimizer=base_optim)
        optimizer = base_optim
        self.optimizer = optimizer
        self.scaler = Scaler(torch.zeros(3))
        self.model.load_state_dict(checkpoint['weights'])
        self.scaler.load_state_dict(checkpoint['scaler_state'])
        self.model_name = checkpoint['model_name']
        print(f"Loaded network '{self.model_name}' from {path}")
