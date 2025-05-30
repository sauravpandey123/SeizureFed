import os
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
import h5py

from dataloader import get_dataloader
from datasets_statistics import *


def get_memory_usage_gb():
    mem_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
    print(f"Memory used: {mem_gb:.2f} GB")


def scale_channel(data, channel_idx, scale_to_nch, nch_channel_index = 7):
    #nch_channel_index = 7 is at F3-C3 since we need the min and max of that channel
    
    if not scale_to_nch:  # Direct standardization
        return (data[:, channel_idx, :][:, np.newaxis, :].astype(np.float32))
    else:  # Scale to NCH range, then standardize
        channel_data = data[:, channel_idx, :].astype(np.float32)  # Ensure precision

        data_min = np.min(channel_data, axis=None)
        data_max = np.max(channel_data, axis=None)

        nch_min, nch_max = nch_min_max(nch_channel_index)
        nch_range = nch_max - nch_min

        eps = 0  # Prevent division by zero

        scaled = ((channel_data - data_min) / (data_max - data_min + eps)) * nch_range + nch_min
            
        return scaled[:, np.newaxis, :].astype(np.float32)
    

def load_patientwise_file(hdf5_path, channel_index, scale_to_nch):
    all_signals, all_labels = [], []
    with h5py.File(hdf5_path, "r") as hf:
        for patient_key in hf:

            signals = hf[patient_key]["signals"][:]
            labels = hf[patient_key]["labels"][:]
            all_signals.append(signals)
            all_labels.append(labels)
    
    all_signals = np.concatenate(all_signals, axis = 0)
    scaled_signals = scale_channel(all_signals, channel_index, scale_to_nch)
    y = np.concatenate(all_labels, axis=0)
    return scaled_signals, y


def standardize_data(scaled, new_mean, new_sd):
        eps = 0
        standardized = (scaled - new_mean) / (new_sd + eps)
        return standardized
    
    
def stratified_train_val_test_split(X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=43):
    assert train_size + val_size + test_size == 1.0, "Splits must sum to 1."

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_size + test_size), random_state=random_state)
    train_idx, temp_idx = next(sss1.split(X, y))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_size / (val_size + test_size), random_state=random_state)
    temp_X, temp_y = X[temp_idx], y[temp_idx]  # Extract temp data
    val_rel_idx, test_rel_idx = next(sss2.split(temp_X, temp_y))

    val_idx = temp_idx[val_rel_idx] 
    test_idx = temp_idx[test_rel_idx]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)