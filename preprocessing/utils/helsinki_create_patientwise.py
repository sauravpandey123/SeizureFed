import os
import h5py
import numpy as np
import tempfile
import shutil

def extract_patient_id(filename):
    return filename.split('-')[0]

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

def get_patientwise_file(input_dir, output_file_dir, output_file_name, batch_size):  
    files = [f for f in os.listdir(input_dir) if f.endswith('.h5')]
    patient_groups = sorted(set(extract_patient_id(f) for f in files))

    temp_output_file = os.path.join(tempfile.gettempdir(), f"{output_file_name}.h5")

    with h5py.File(temp_output_file, "w") as hf_out:
        for patient_id in patient_groups:
            patient_files = sorted([f for f in files if extract_patient_id(f) == patient_id])
            patient_group = hf_out.require_group(patient_id)

            if "signals" not in patient_group:
                dset_signals = patient_group.create_dataset(
                    "signals", shape=(0, 12, 256), maxshape=(None, 12, 256),
                    chunks=(batch_size, 12, 256), compression="gzip", compression_opts=9, dtype='float64'
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

                    assert batch_signals.shape[0] == batch_labels.shape[0], \
                        f"Mismatch in shapes for file {file}: {batch_signals.shape[0]} vs {batch_labels.shape[0]}"

                    append_to_dataset(dset_signals, batch_signals)
                    append_to_dataset(dset_labels, batch_labels)

                print("Processed:", file)

    os.makedirs(output_file_dir, exist_ok=True)
    final_output_path = os.path.join(output_file_dir, f"{output_file_name}.h5")

    try:
        shutil.move(temp_output_file, final_output_path)
        print("Final .h5 file saved to:", final_output_path)
    except Exception as e:
        print("Failed to move .h5 file to output directory:", e)
