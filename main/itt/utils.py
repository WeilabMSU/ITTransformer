import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump
from torch.optim.lr_scheduler import _LRScheduler


# --- Element and Cluster Information ---
class ElementClusterInfo(object):
    atomic_number_to_symbol = {
        "1": "H", "2": "He", "3": "Li", "4": "Be", "5": "B", "6": "C", "7": "N", "8": "O", "9": "F", "10": "Ne", "11": "Na", "12": "Mg", "13": "Al", "14": "Si", "15": "P", "16": "S", "17": "Cl", "18": "Ar", "19": "K", "20": "Ca", "21": "Sc", "22": "Ti", "23": "V", "24": "Cr", "25": "Mn", "26": "Fe", "27": "Co", "28": "Ni", "29": "Cu", "30": "Zn", "31": "Ga", "32": "Ge", "33": "As", "34": "Se", "35": "Br", "36": "Kr", "37": "Rb", "38": "Sr", "39": "Y", "40": "Zr", "41": "Nb", "42": "Mo", "43": "Tc", "44": "Ru", "45": "Rh", "46": "Pd", "47": "Ag", "48": "Cd", "49": "In", "50": "Sn", "51": "Sb", "52": "Te", "53": "I", "54": "Xe", "55": "Cs", "56": "Ba", "57": "La", "58": "Ce", "59": "Pr", "60": "Nd", "61": "Pm", "62": "Sm", "63": "Eu", "64": "Gd", "65": "Tb", "66": "Dy", "67": "Ho", "68": "Er", "69": "Tm", "70": "Yb", "71": "Lu", "72": "Hf", "73": "Ta", "74": "W", "75": "Re", "76": "Os", "77": "Ir", "78": "Pt", "79": "Au", "80": "Hg", "81": "Tl", "82": "Pb", "83": "Bi", "84": "Po", "85": "At", "86": "Rn", "87": "Fr", "88": "Ra", "89": "Ac", "90": "Th", "91": "Pa", "92": "U", "93": "Np", "94": "Pu", "95": "Am", "96": "Cm", "97": "Bk", "98": "Cf", "99": "Es", "100": "Fm", "101": "Md", "102": "No", "103": "Lr", "104": "Rf", "105": "Db", "106": "Sg", "107": "Bh", "108": "Hs", "109": "Mt", "110": "Ds", "111": "Rg", "112": "Cn", "113": "Nh", "114": "Fl", "115": "Mc", "116": "Lv", "117": "Ts", "118": "Og"
    }
    symbol_to_atomic_number = {
        "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
    }
    symbol_to_cluster = {
        "H": 1, "He": 4, "Li": 6, "Be": 4, "B": 0, "C": 0, "N": 3, "O": 2, "F": 5, "Ne": 4, "Na": 6, "Mg": 6, "Al": 4, "Si": 0, "P": 0, "S": 5, "Cl": 5, "Ar": 4, "K": 6, "Ca": 6, "Sc": 4, "Ti": 4, "V": 4, "Cr": 4, "Mn": 4, "Fe": 4, "Co": 4, "Ni": 4, "Cu": 4, "Zn": 4, "Ga": 4, "Ge": 5, "As": 5, "Se": 5, "Br": 5, "Kr": 4, "Rb": 6, "Sr": 6, "Y": 4, "Zr": 4, "Nb": 4, "Mo": 4, "Tc": 4, "Ru": 4, "Rh": 4, "Pd": 4, "Ag": 4, "Cd": 4, "In": 4, "Sn": 4, "Sb": 4, "Te": 5, "I": 5, "Xe": 4, "Cs": 6, "Ba": 6, "La": 4, "Ce": 4, "Pr": 4, "Nd": 4, "Pm": 4, "Sm": 4, "Eu": 4, "Gd": 4, "Tb": 4, "Dy": 4, "Ho": 4, "Er": 4, "Tm": 4, "Yb": 4, "Lu": 4, "Hf": 4, "Ta": 4, "W": 4, "Re": 4, "Os": 4, "Ir": 4, "Pt": 4, "Au": 4, "Hg": 4, "Tl": 4, "Pb": 4, "Bi": 4, "Po": 4, "At": 4, "Rn": 4, "Fr": 4, "Ra": 4, "Ac": 4, "Th": 4, "Pa": 4, "U": 4, "Np": 4, "Pu": 4, "Am": 4, "Cm": 4, "Bk": 4, "Cf": 4, "Es": 4, "Fm": 4, "Md": 4, "No": 4, "Lr": 4, "Rf": 4, "Db": 4, "Sg": 4, "Bh": 4, "Hs": 4, "Mt": 4, "Ds": 4, "Rg": 4, "Cn": 4, "Nh": 4, "Fl": 4, "Mc": 4, "Lv": 4, "Ts": 4, "Og": 4
    }
    atomic_number_to_cluster = {
        "1": 1, "2": 4, "3": 6, "4": 4, "5": 0, "6": 0, "7": 3, "8": 2, "9": 5, "10": 4, "11": 6, "12": 6, "13": 4, "14": 0, "15": 0, "16": 5, "17": 5, "18": 4, "19": 6, "20": 6, "21": 4, "22": 4, "23": 4, "24": 4, "25": 4, "26": 4, "27": 4, "28": 4, "29": 4, "30": 4, "31": 4, "32": 5, "33": 5, "34": 5, "35": 5, "36": 4, "37": 6, "38": 6, "39": 4, "40": 4, "41": 4, "42": 4, "43": 4, "44": 4, "45": 4, "46": 4, "47": 4, "48": 4, "49": 4, "50": 4, "51": 4, "52": 5, "53": 5, "54": 4, "55": 6, "56": 6, "57": 4, "58": 4, "59": 4, "60": 4, "61": 4, "62": 4, "63": 4, "64": 4, "65": 4, "66": 4, "67": 4, "68": 4, "69": 4, "70": 4, "71": 4, "72": 4, "73": 4, "74": 4, "75": 4, "76": 4, "77": 4, "78": 4, "79": 4, "80": 4, "81": 4, "82": 4, "83": 4, "84": 4, "85": 4, "86": 4, "87": 4, "88": 4, "89": 4, "90": 4, "91": 4, "92": 4, "93": 4, "94": 4, "95": 4, "96": 4, "97": 4, "98": 4, "99": 4, "100": 4, "101": 4, "102": 4, "103": 4, "104": 4, "105": 4, "106": 4, "107": 4, "108": 4, "109": 4, "110": 4, "111": 4, "112": 4, "113": 4, "114": 4, "115": 4, "116": 4, "117": 4, "118": 4
        }
    # Electronegativity values (Pauling scale)
    atomic_number_to_electronegativity = {
        1: 2.20, 2: 0.00, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 10: 0.00,
        11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58, 17: 3.16, 18: 0.00, 19: 0.82, 20: 1.00,
        21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
        31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 0.00, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33,
        41: 1.6, 42: 2.16, 43: 1.9, 44: 2.2, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96,
        51: 2.05, 52: 2.1, 53: 2.66, 54: 0.00, 55: 0.79, 56: 0.89, 57: 1.10, 58: 1.12, 59: 1.13, 60: 1.14,
        61: 1.13, 62: 1.17, 63: 1.2, 64: 1.2, 65: 1.2, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 70: 1.1,
        71: 1.27, 72: 1.3, 73: 1.5, 74: 2.36, 75: 1.9, 76: 2.2, 77: 2.2, 78: 2.28, 79: 2.54, 80: 2.00,
        81: 1.62, 82: 1.87, 83: 2.02, 84: 2.0, 85: 2.2, 86: 0.00, 87: 0.7, 88: 0.9, 89: 1.1, 90: 1.3,
        91: 1.5, 92: 1.38, 93: 1.36, 94: 1.28, 95: 1.3, 96: 1.3, 97: 1.3, 98: 1.3, 99: 1.3, 100: 1.3,
        101: 1.3, 102: 1.3, 103: 1.3, 104: 0.00, 105: 0.00, 106: 0.00, 107: 0.00, 108: 0.00, 109: 0.00, 110: 0.00,
        111: 0.00, 112: 0.00, 113: 0.00, 114: 0.00, 115: 0.00, 116: 0.00, 117: 0.00, 118: 0.00
    }
    total_element_cluster = 7

    @classmethod
    def get_cluster_from_symbol(cls, symbol: str):
        return cls.symbol_to_cluster.get(symbol)

    @classmethod
    def get_cluster_from_atomic_number(cls, atomic_number: str):
        return cls.atomic_number_to_cluster.get(atomic_number)

    @classmethod
    def get_symbol_from_atomic_number(cls, atomic_number: str):
        return cls.atomic_number_to_symbol.get(atomic_number)

    @classmethod
    def get_atomic_number_from_symbol(cls, symbol: str):
        return cls.symbol_to_atomic_number.get(symbol)
    
    @classmethod
    def get_electronegativity_from_atomic_number(cls, atomic_number: int):
        """Get electronegativity value for a given atomic number."""
        return cls.atomic_number_to_electronegativity.get(atomic_number, 0.0)
    
    @classmethod
    def get_electronegativity_from_symbol(cls, symbol: str):
        """Get electronegativity value for a given element symbol."""
        atomic_number = cls.get_atomic_number_from_symbol(symbol)
        if atomic_number is not None:
            return cls.get_electronegativity_from_atomic_number(atomic_number)
        return 0.0

# --- Metrics Calculation ---
def calculate_metrics(y_true, y_pred):
    """Calculates a dictionary of regression metrics."""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    pearson_r, _ = pearsonr(y_true, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_r2': pearson_r**2
    }

def calculate_classification_metrics(y_true, y_pred):
    """Calculates a dictionary of classification metrics."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Convert predictions to class labels
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        # Multi-class: take argmax
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        # Binary: threshold at 0.5
        y_pred_labels = (y_pred.flatten() > 0.5).astype(int)
    
    # Convert true labels to int if needed
    y_true_labels = y_true.flatten().astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    
    # For binary classification, calculate additional metrics
    if len(np.unique(y_true_labels)) == 2:
        precision = precision_score(y_true_labels, y_pred_labels, average='binary', zero_division=0)
        recall = recall_score(y_true_labels, y_pred_labels, average='binary', zero_division=0)
        f1 = f1_score(y_true_labels, y_pred_labels, average='binary', zero_division=0)
    else:
        # Multi-class: use macro average
        precision = precision_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
        recall = recall_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
        f1 = f1_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# --- Helper Classes for Graph Feature Generation (inspired by CGCNN) ---
class CustomAtomInitializer(object):
    """
    Initialize atom features from a JSON file.
    """
    def __init__(self):
        script_dir = os.path.dirname(__file__)
        atom_init_path = os.path.join(script_dir, 'atom_init.json')
        print(f"atom_init_path: {atom_init_path}")
        assert os.path.exists(atom_init_path), f"atom_init.json not found at {atom_init_path}"

        with open(atom_init_path) as f:
            elem_embedding = json.load(f)
        self.elem_embedding = {int(key): value for key, value
                               in elem_embedding.items()}
        atom_types = set(self.elem_embedding.keys())
        self.atom_types = list(atom_types)
        self.n_features = len(self.elem_embedding[self.atom_types[0]])

    def get_atom_fea(self, atom_no):
        return np.array(self.elem_embedding[atom_no])


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.
    """
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class IndividualFeatureDataset:
    def __init__(self, split_data, features_folder, feature_scaler=None, radius=8.0, max_num_nbr=12):
        self.indices, self.file_ids, self.labels = split_data
        self.features_folder = features_folder
        self.feature_scaler = feature_scaler

        self.atom_initializer = CustomAtomInitializer()
        self.dist_converter = GaussianDistance(dmin=0, dmax=radius, step=0.2)
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.labels[idx]
        
        # Load individual feature file
        feature_file = os.path.join(self.features_folder, f"{file_id}.npz")
        data = np.load(feature_file, allow_pickle=True)
        
        # Extract features
        part0 = data['part0']
        part1 = data['part1']
        part2 = data['part2']
        part3_atom_nos = data['part3_atom_nos']
        part3_nbr_fea_dist = data['part3_nbr_fea_dist']
        part3_nbr_fea_idx = data['part3_nbr_fea_idx']
        part3_cluster_indices = data['part3_cluster_indices']

        if self.feature_scaler is not None:
            part0 = self.feature_scaler[0].transform(part0.reshape(-1, part0.shape[-1])).reshape(part0.shape)
            part1 = self.feature_scaler[1].transform(part1.reshape(-1, part1.shape[-1])).reshape(part1.shape)
            part2 = self.feature_scaler[2].transform(part2.reshape(-1, part2.shape[-1])).reshape(part2.shape)

        atom_fea = np.vstack([self.atom_initializer.get_atom_fea(no) for no in part3_atom_nos])
        nbr_fea = self.dist_converter.expand(part3_nbr_fea_dist.astype(np.float32)/1000)
        nbr_fea_idx = part3_nbr_fea_idx.astype(int)
        cluster_indices = part3_cluster_indices

        return (part0, part1, part2, (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices)), label


def collate_fn(dataset_list):
    batch_part0, batch_part1, batch_part2, batch_labels = [], [], [], []
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, batch_cluster_indices = [], [], [], []
    batch_cluster_indices = []
    crystal_atom_idx, base_idx = [], 0
    
    for i, (data_point, label) in enumerate(dataset_list):
        part0, part1, part2, (atom_fea, nbr_fea, nbr_fea_idx, cluster_indices) = data_point
        
        batch_part0.append(torch.from_numpy(part0))
        batch_part1.append(torch.from_numpy(part1))
        batch_part2.append(torch.from_numpy(part2))
        batch_labels.append(torch.tensor(label, dtype=torch.float32))
        
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(torch.from_numpy(atom_fea))
        batch_nbr_fea.append(torch.from_numpy(nbr_fea))
        batch_nbr_fea_idx.append(torch.from_numpy(nbr_fea_idx) + base_idx)
        batch_cluster_indices.append(torch.from_numpy(cluster_indices))
        
        new_idx = torch.LongTensor(np.arange(n_i)) + base_idx
        crystal_atom_idx.append(new_idx)
        base_idx += n_i

    return (torch.stack(batch_part0, dim=0),
            torch.stack(batch_part1, dim=0),
            torch.stack(batch_part2, dim=0),
            (torch.cat(batch_atom_fea, dim=0).float(),
             torch.cat(batch_nbr_fea, dim=0).float(),
             torch.cat(batch_nbr_fea_idx, dim=0).long(),
             torch.cat(batch_cluster_indices, dim=0).long(),
             crystal_atom_idx)), \
           torch.stack(batch_labels, dim=0)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return None

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return None


def fit_and_save_feature_scalers(feature_folder, scaler_save_path):
    """
    Fit StandardScaler for part0, part1, and part2 from all .npz files in the folder and save as a tuple.
    Args:
        feature_folder (str): Path to folder containing .npz feature files (one per sample).
        scaler_save_path (str): Path to save the tuple of scalers (joblib file).
    """
    part0_list, part1_list, part2_list = [], [], []
    npz_files = [f for f in os.listdir(feature_folder) if f.endswith('.npz')]
    for fname in npz_files:
        data = np.load(os.path.join(feature_folder, fname), allow_pickle=True)
        part0_list.append(data['part0'].reshape(-1, data['part0'].shape[-1]))
        part1_list.append(data['part1'].reshape(-1, data['part1'].shape[-1]))
        part2_list.append(data['part2'].reshape(-1, data['part2'].shape[-1]))
    part0_all = np.vstack(part0_list)
    part1_all = np.vstack(part1_list)
    part2_all = np.vstack(part2_list)
    part0_scaler = StandardScaler().fit(part0_all)
    part1_scaler = StandardScaler().fit(part1_all)
    part2_scaler = StandardScaler().fit(part2_all)
    scalers = (part0_scaler, part1_scaler, part2_scaler)
    dump(scalers, scaler_save_path)
    print(f"Feature scalers saved to: {scaler_save_path}")
    return scaler_save_path


def fit_and_save_feature_scaler_tar(tar_path, scaler_save_path):
    """
    Fit StandardScaler for part0, part1, and part2 from all .npz files in a tar archive and save as a tuple.
    Args:
        tar_path (str): Path to .tar file containing .npz feature files.
        scaler_save_path (str): Path to save the tuple of scalers (joblib file).
    """
    import tarfile
    
    part0_list, part1_list, part2_list = [], [], []
    
    with tarfile.open(tar_path, 'r') as tar:
        # Get all .npz files in the tar
        npz_members = [m for m in tar.getmembers() if m.name.endswith('.npz')]
        
        print(f"Found {len(npz_members)} .npz files in tar archive")
        
        for member in npz_members:
            f = tar.extractfile(member)
            data = np.load(f, allow_pickle=True)
            part0_list.append(data['part0'].reshape(-1, data['part0'].shape[-1]))
            part1_list.append(data['part1'].reshape(-1, data['part1'].shape[-1]))
            part2_list.append(data['part2'].reshape(-1, data['part2'].shape[-1]))
    
    part0_all = np.vstack(part0_list)
    part1_all = np.vstack(part1_list)
    part2_all = np.vstack(part2_list)
    
    part0_scaler = StandardScaler().fit(part0_all)
    part1_scaler = StandardScaler().fit(part1_all)
    part2_scaler = StandardScaler().fit(part2_all)
    
    scalers = (part0_scaler, part1_scaler, part2_scaler)
    dump(scalers, scaler_save_path)
    print(f"Feature scalers saved to: {scaler_save_path}")
    return scaler_save_path


def fit_and_save_feature_scaler_tar_incremental(tar_path, scaler_save_path):
    """
    Incrementally fit StandardScaler for part0, part1, and part2 from all .npz files in a tar archive
    and save the scalers as a tuple.
    """
    import tarfile

    part0_scaler = StandardScaler()
    part1_scaler = StandardScaler()
    part2_scaler = StandardScaler()
    
    count = 0

    with tarfile.open(tar_path, 'r') as tar:
        npz_members = [m for m in tar.getmembers() if m.name.endswith('.npz')]
        print(f"Found {len(npz_members)} .npz files in tar archive")

        for member in npz_members:
            f = tar.extractfile(member)
            if f is None:
                continue
            data = np.load(f, allow_pickle=True)

            # Reshape and fit incrementally
            part0 = data['part0'].reshape(-1, data['part0'].shape[-1])
            part1 = data['part1'].reshape(-1, data['part1'].shape[-1])
            part2 = data['part2'].reshape(-1, data['part2'].shape[-1])

            part0_scaler.partial_fit(part0)
            part1_scaler.partial_fit(part1)
            part2_scaler.partial_fit(part2)
            count += 1

            if count % 1000 == 0:
                print(f"Processed {count} files...")

    scalers = (part0_scaler, part1_scaler, part2_scaler)
    dump(scalers, scaler_save_path)
    print(f"Feature scalers saved to: {scaler_save_path}")
    return scaler_save_path


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup and linear decay learning rate scheduler.
    Mimics the behavior of Hugging Face's get_linear_schedule_with_warmup.
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * float(self.last_epoch) / float(max(1, self.num_warmup_steps))
                for base_lr in self.base_lrs
            ]
        else:
            # Linear decay
            decay_steps = self.num_training_steps - self.num_warmup_steps
            decay_progress = float(self.last_epoch - self.num_warmup_steps) / float(max(1, decay_steps))
            return [
                base_lr * max(0.0, 1.0 - decay_progress)
                for base_lr in self.base_lrs
            ]

