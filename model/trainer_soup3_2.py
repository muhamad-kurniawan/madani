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
    """Trainer with shared optimiser defaults, LR‑peak logging, and a safe
    `.reshape()` use to avoid non‑contiguous view errors."""

    # ────────────────────────────────────────────────────
    #  Init
    # ────────────────────────────────────────────────────
    def __init__(self, model, model_name: str = "UnnamedModel", n_elements: int = 8,
                 capture_every=None, verbose: bool = True, drop_unary: bool = True,
                 scale: bool = True, default_optim_params: dict | None = None):

        self.model = model
        self.model_name = model_name
        self.n_elements = n_elements
        self.capture_every = capture_every
        self.verbose = verbose
        self.drop_unary = drop_unary
        self.scale = scale
        self.classification = False

        self.default_optim_params = default_optim_params or {
            "base_lr": 1e-4,
            "max_lr": 6e-3,
            "lr_decay": 1.0,
            "epochs_step": 1,
        }

        # misc helpers
        self.fudge = 0.02
        self.compute_device = getattr(model, "compute_device", None) or get_compute_device()

        # attention capture
        self.capture_flag = False
        self.formula_current = None
        self.act_v = None
        self.pred_v = None

        if verbose:
            print("\nModel architecture: out_dims, d_model, N, heads")
            print(f"{model.out_dims}, {model.d_model}, {model.N}, {model.heads}")
            print(f"Running on compute device: {self.compute_device}")
            print(f"Model size: {count_parameters(model)} parameters\n")
            if capture_every is not None:
                print(f"Capturing attention tensors every {capture_every}")

    # ────────────────────────────────────────────────────
    #  Helpers
    # ────────────────────────────────────────────────────
    def _merge_optim(self, override):
        return {**self.default_optim_params, **(override or {})}

    def set_default_optim_params(self, **kwargs):
        self.default_optim_params.update(kwargs)
        if self.verbose:
            print("[Trainer] Default optimiser params now:", self.default_optim_params)

    # ────────────────────────────────────────────────────
    #  Data loading
    # ────────────────────────────────────────────────────
    def load_data(self, file_name: str, *, batch_size: int = 512, train: bool = False):
        self.batch_size = batch_size
        handler = DataHandler(file_name, batch_size=batch_size, n_elements=self.n_elements,
                              inference=not train, verbose=self.verbose,
                              drop_unary=self.drop_unary, scale=self.scale)
        if self.verbose:
            print(f"Loading data with up to {handler.n_elements:0.0f} elements in the formula")
        self.n_elements = handler.n_elements
        self.data_loader = handler.get_data_loaders(inference=not train)
        if train:
            y = self.data_loader.dataset.y
            self.train_loader = self.data_loader
            self.scaler = DummyScaler(y) if self.classification else Scaler(y)

    # ────────────────────────────────────────────────────
    #  Training phase (uses .reshape to avoid view errors)
    # ────────────────────────────────────────────────────
    def _train_phase(self, label: str, *, epochs: int, checkin: int, optim_params: dict | None):
        opt = self._merge_optim(optim_params)
        print(f"\n— {label}: {epochs} epochs — optimiser {opt}")

        # criterion & optimiser
        self.criterion = RobustL1 if not self.classification else BCEWithLogitsLoss
        self.optimizer = Lamb(self.model.parameters())

        base_lr, max_lr = opt["base_lr"], opt["max_lr"]
        lr_decay, epochs_step = opt["lr_decay"], opt["epochs_step"]
        step_up = epochs_step * len(self.train_loader)

        def cy_sched(lo, hi):
            return torch.optim.lr_scheduler.CyclicLR(
                self.optimizer, base_lr=lo, max_lr=hi,
                step_size_up=step_up, cycle_momentum=False)

        self.lr_scheduler = cy_sched(base_lr, max_lr)
        best_mae, best_state = float("inf"), copy.deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            self.model.train(); peak_announced = False

            for batch_idx, (X, y, _) in enumerate(self.train_loader):
                y = self.scaler.scale(y)
                src, frac = X[:, :, 0], X[:, :, 1]
                frac = torch.clamp(frac * (1 + torch.randn_like(frac) * self.fudge), 0, 1)
                frac[src == 0] = 0; frac = frac / frac.sum(dim=1, keepdim=True)
                src  = src.to(self.compute_device, dtype=torch.long)
                frac = frac.to(self.compute_device, dtype=data_type_torch)
                y    = y.to(self.compute_device, dtype=data_type_torch)

                pred, log_u = self.model(src, frac).chunk(2, dim=-1)
                # — use reshape to handle non‑contiguous tensors —
                loss = self.criterion(pred.reshape(-1), log_u.reshape(-1), y.reshape(-1))

                lr_before = self.optimizer.param_groups[0]['lr']
                loss.backward(); self.optimizer.step(); self.optimizer.zero_grad()
                self.lr_scheduler.step()
                lr_after = self.optimizer.param_groups[0]['lr']

                if not peak_announced and lr_after < lr_before:
                    print(f"        LR peak reached at epoch {epoch} (batch {batch_idx}) — lr={lr_before:.3e}")
                    peak_announced = True

            # validation
            if (epoch + 1) % checkin == 0 or epoch in (0, epochs - 1):
                mae = mean_absolute_error(*self.predict(self.data_loader)[:2])
                print(f"    Epoch {epoch:03d} — val MAE={mae:.4g}")
                if mae < best_mae:
                    best_mae, best_state = mae, copy.deepcopy(self.model.state_dict())
                    print("      ↳ new best")

            # decay LR envelope
            if lr_decay < 1.0:
                base_lr *= lr_decay; max_lr *= lr_decay
                self.lr_scheduler = cy_sched(base_lr, max_lr)

        self.model.load_state_dict(best_state)
        print(f"Loaded best {label} model (MAE={best_mae:.4g})")

    # ────────────────────────────────────────────────────
    #  Public wrappers
    # ────────────────────────────────────────────────────
    def pretrain(self, *, epochs: int = 10, checkin: int = 2, optim_params: dict | None = None):
        self._train_phase("Pretrain", epochs=epochs, checkin=checkin, optim_params=optim_params)

    def finetune(self, *, epochs: int = 20, checkin: int = 2, optim_params: dict | None = None):
        self._train_phase("Finetune", epochs=epochs, checkin=checkin, optim_params=optim_params)

    # ────────────────────────────────────────────────────
    #  Soup helper (plain average)
    # ────────────────────────────────────────────────────
    def _run_variant(self, override, *, epochs, checkin):
        start_state = copy.deepcopy(self.model.state_dict())
        self._train_phase("Variant", epochs=epochs, checkin=checkin, optim_params=override)
        mae = mean_absolute_error(*self.predict(self.data_loader)[:2])
        state = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(start_state)
        return state, mae

    def finetune_variants_for_soup(self, variant_configs: list | None = None, *, epochs: int = 20, checkin: int = 2):
        variant_configs = variant_configs or [None]
        pretrained_state = copy.deepcopy(self.model.state_dict())
        states = []
        for i, cfg in enumerate(variant_configs):
            print(f"\n--- Variant {i+1}/{len(variant_configs)} ---")
            self.model.load_state_dict(pretrained_state)
            st, _ = self._run_variant(cfg, epochs=epochs, checkin=checkin)
            states.append(st)
        avg_state = {k: sum(st[k] for st in states) / len(states) for k in states[0]}
        self.model.load_state_dict(avg_state)
        print(f"\n[Plain Soup] Averaged {len(states)} variant(s) — loaded into model")
      
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
