import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import OrderedDict, Counter
from tqdm import tqdm
import re

data_type_np = np.float32
data_type_torch = torch.float32

def _element_composition(formula):

    # e.g. "NaCl" -> {"Na": 1, "Cl": 1}
    # e.g. "SiO2" -> {"Si": 1, "O": 2}
    
    pattern = r'([A-Z][a-z]?)(\d*)'
    tokens = re.findall(pattern, formula)
    comp = Counter()
    for elem, num in tokens:
        if num == '':
            comp[elem] += 1
        else:
            comp[elem] += int(num)
    return dict(comp)


def parse_csv(data,
                  n_elements=6,
                  drop_unary=True,
                  scale=True,
                  inference=False,
                  verbose=True):
   
    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(data, keep_default_na=False, na_values=[''])
    else:
        df = data

    if 'formula' not in df.columns:
        df['formula'] = df['cif_id'].str.split('_ICSD').str[0]

    df['count'] = [len(_element_composition(form)) for form in df['formula']]
    # Optionally drop unary (single-element) compounds
    if drop_unary:
        df = df[df['count'] != 1]

    # If not inference, group duplicates by formula and average them
    if not inference:
        df = df.groupby('formula').mean().reset_index()

    list_ohm = [OrderedDict(_element_composition(form)) for form in df['formula']]

    y = df['target'].values.astype(data_type_np)
    formula = df['formula'].values

    N = len(list_ohm)
    edm_array = np.zeros((N, n_elements, len(all_symbols) + 1), dtype=data_type_np)
    elem_num = np.zeros((N, n_elements), dtype=data_type_np)
    elem_frac = np.zeros((N, n_elements), dtype=data_type_np)

    # Fill in the arrays
    iterator = tqdm(list_ohm, desc="Generating EDM", unit="formulae", disable=not verbose)
    for i, comp in enumerate(iterator):
        # comp is an OrderedDict of {elem: count, ...} for a single formula
        for j, (elem, count) in enumerate(comp.items()):
            if j == n_elements:  # Truncate if there are more than n_elements
                break
            try:
                idx_symbol = all_symbols.index(elem) + 1
                edm_array[i, j, idx_symbol] = count
                elem_num[i, j] = idx_symbol
            except ValueError:
                # If the element isn't in all_symbols, skip
                pass

    for i in range(N):
        # sum across the 'symbol' dimension
        raw_counts = edm_array[i, :, :].sum(axis=-1)
        if scale:
            total = raw_counts.sum()
            if total > 0:
                elem_frac[i, :] = raw_counts / total
        else:
            elem_frac[i, :] = raw_counts

    elem_num = elem_num.reshape(N, n_elements, 1)
    elem_frac = elem_frac.reshape(N, n_elements, 1)
    X = np.concatenate((elem_num, elem_frac), axis=-1)  # shape [N, n_elements, 2]

    return X, y, formula


class MaterialDataset(Dataset):
    """
    A custom PyTorch Dataset for EDM data.
    Expects X, y, and formula as numpy arrays.
    """

    def __init__(self, X, y, formula, transform=None):
        """
        Parameters
        ----------
        X : np.ndarray, shape = [N, n_elements, 2]
        y : np.ndarray, shape = [N]
        formula : np.ndarray, shape = [N]
        transform : callable, optional
            An optional transform to apply to X or (X, y).
        """
        self.X = X
        self.y = y
        self.formula = formula
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_sample = self.X[idx]
        y_sample = self.y[idx]
        formula_str = self.formula[idx]

        # Convert X and y to tensors
        X_tensor = torch.as_tensor(X_sample, dtype=data_type_torch)
        y_tensor = torch.as_tensor(y_sample, dtype=data_type_torch)

        # Apply an optional transform
        if self.transform is not None:
            X_tensor, y_tensor = self.transform(X_tensor, y_tensor)

        return X_tensor, y_tensor, formula_str


class DataHandler:

    def __init__(self,
                 csv_file,
                 n_elements=6,
                 drop_unary=True,
                 scale=True,
                 inference=False,
                 verbose=True,
                 batch_size=64,
                 val_split=0.2,
                 random_seed=42,
                 shuffle=True,
                 pin_memory=True):
        """
        Parameters
        ----------
        csv_file : str or pd.DataFrame
            Path to CSV or DataFrame containing your data.
        n_elements : int
            Number of elements to retain in EDM representation.
        drop_unary : bool
            Drop single-element formulas.
        scale : bool
            Normalize or not.
        inference : bool
            If True, skip groupby averaging and splitting.
        verbose : bool
            Whether to show progress bars.
        batch_size : int
            Batch size for each DataLoader.
        val_split : float
            Fraction of data to use as validation.
        random_seed : int
            Seed for splitting reproducibility.
        shuffle : bool
            Shuffle the data or not.
        pin_memory : bool
            DataLoader pin_memory argument.
        """

        # Parse the data
        X, y, formula = parse_csv(csv_file,
                                      n_elements=n_elements,
                                      drop_unary=drop_unary,
                                      scale=scale,
                                      inference=inference,
                                      verbose=verbose)
        self.inference = inference

        # Build a dataset
        self.dataset = MaterialDataset(X, y, formula)

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.random_seed = random_seed
        self.val_split = val_split
        self.data_len = len(self.dataset)

    def get_data_loaders(self):
        """
        Returns one or two DataLoader objects. If `inference=True`, returns only one.
        Otherwise, returns (train_loader, val_loader).
        """

        if self.inference:
            # For inference, just return the entire dataset in one DataLoader
            loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=False,  # Usually don't shuffle inference data
                pin_memory=self.pin_memory
            )
            return loader

        # Otherwise, do a train/val split
        val_size = int(self.val_split * self.data_len)
        train_size = self.data_len - val_size

        # Reproducible split
        generator = torch.Generator().manual_seed(self.random_seed)
        train_dataset, val_dataset = random_split(self.dataset,
                                                 [train_size, val_size],
                                                 generator=generator)
        # Build loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory
        )
        return train_loader, val_loader
