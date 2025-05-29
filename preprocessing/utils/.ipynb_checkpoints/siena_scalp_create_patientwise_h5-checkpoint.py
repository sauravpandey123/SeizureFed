import os
import h5py
import numpy as np
from tqdm import tqdm
import tempfile
import shutil

def extract_patient_id(filename):
    return filename.split('.')[0]

def load_signal_and_label(file_path):
    with h5py.File(file_path, "r") as f:
        signals = f["signals"][:].astype(np.float16)
        labels = f["labels"][:].astype(np.int8)
    return signals, labels

def append_to_dataset(dataset, data):
    current_size = dataset.shape[0]
    new_size = current_size + data.shape[0]
    dataset.resize((new_size,) + dataset.shape[1:])
    dataset[current_size:new_size] = data

def create_patientwise_h5(individual_h5_files, output_h5_file, output_file_dir, batch_size):
    input_dir = individual_h5_files
    os.makedirs(output_file_dir, exist_ok=True)

    # Create a temp file path to avoid NFS locking issues
    temp_output_path = os.path.join(tempfile.gettempdir(), f"{output_h5_file}.h5")

    files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    patient_groups = sorted(set(extract_patient_id(f) for f in files))

    with h5py.File(temp_output_path, "w") as hf_out:
        for patient_id in patient_groups:
            print("Processing Patient:", patient_id)
            patient_files = sorted([f for f in files if extract_patient_id(f) == patient_id])

            patient_group = hf_out.require_group(patient_id)

            if "signals" not in patient_group:
                dset_signals = patient_group.create_dataset(
                    "signals", shape=(0, 10, 256), maxshape=(None, 10, 256),
                    chunks=(batch_size, 10, 256), compression="gzip", compression_opts=9, dtype='float16'
                )
                dset_labels = patient_group.create_dataset(
                    "labels", shape=(0,), maxshape=(None,),
                    chunks=(batch_size,), compression="gzip", compression_opts=9, dtype='int8'
                )
            else:
                dset_signals = patient_group["signals"]
                dset_labels = patient_group["labels"]

            for file in patient_files:
                file_path = os.path.join(input_dir, file)
                signals, labels = load_signal_and_label(file_path)
                for start_idx in range(0, signals.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, signals.shape[0])
                    batch_signals = signals[start_idx:end_idx]
                    batch_labels = labels[start_idx:end_idx]

                    assert batch_signals.shape[0] == batch_labels.shape[0], "Mismatch between signals and labels count."

                    append_to_dataset(dset_signals, batch_signals)
                    append_to_dataset(dset_labels, batch_labels)

    final_output_path = os.path.join(output_file_dir, f"{output_h5_file}.h5")

    try:
        shutil.move(temp_output_path, final_output_path)
        print(f"\nData saved successfully to '{final_output_path}'")
    except Exception as e:
        print(f"\nFailed to move final HDF5 file: {e}")
