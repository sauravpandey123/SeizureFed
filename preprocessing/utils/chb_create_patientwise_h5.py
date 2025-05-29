import numpy as np
import os
import h5py
import gc
import tempfile
import shutil

from utils.chb_preprocessing import *
from utils.signal_augmentation import augment_seizure_cases


def extract_patient_id(file_name):
    return file_name.split('_')[0]


def get_patientwise_file(individual_files, output_file_dir, output_file_name):
    myfiles = os.listdir(individual_files)
    os.makedirs(output_file_dir, exist_ok=True)

    channels = 19
    time_steps = 256
    chunk_size = 25

    patch_intervals = 2
    preictal_period = 60
    postictal_period = 10

    temp_output_file = os.path.join(tempfile.gettempdir(), f"{output_file_name}.h5")

    with h5py.File(temp_output_file, "w") as hf_out:
        for file in myfiles:
            patient_id = extract_patient_id(file)

            if patient_id == 'chb24':
                print("Skipping chb24 as it is not used...")
                continue

            print(f"\nProcessing {file} for {patient_id}...")

            try:
                with h5py.File(os.path.join(individual_files, file), 'r') as hf_in:
                    y = hf_in['data'][:]
            except Exception as e:
                print(f"Error opening {file}: {e}")
                continue

            total_samples_per_patch = 256 * patch_intervals
            total_patches = y.shape[0] // total_samples_per_patch

            data = y[:total_patches * total_samples_per_patch, :].reshape(
                total_patches, total_samples_per_patch, -1
            )
            final_data = data.transpose(0, 2, 1)

            labels = np.zeros(final_data.shape[0], dtype=int)
            final_labels = get_filled_labels(final_data, labels)
            shortened_data, shortened_labels = get_labels_and_data(
                final_data, final_labels, patch_intervals, preictal_period, postictal_period
            )

            downsampled_data = batch_process_downsampling(
                shortened_data, batch_size=50, cutoff=64, fs=256, order=2
            )

            default_overlap = {0: 0, 1: 0.6}
            aug_signals, aug_labels, _ = augment_seizure_cases(
                downsampled_data, shortened_labels, default_overlap, auto_balance=True
            )

            aug_signals = aug_signals.astype(np.float16)
            aug_labels = aug_labels.astype(np.int8)

            patient_group = hf_out.require_group(patient_id)

            if 'signals' not in patient_group:
                dset_signals = patient_group.create_dataset(
                    "signals",
                    shape=(0, channels, time_steps),
                    maxshape=(None, channels, time_steps),
                    chunks=(chunk_size, channels, time_steps),
                    compression="gzip",
                    compression_opts=9,
                    dtype='float16'
                )
                dset_labels = patient_group.create_dataset(
                    "labels",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(chunk_size,),
                    compression="gzip",
                    compression_opts=9,
                    dtype='int8'
                )
            else:
                dset_signals = patient_group['signals']
                dset_labels = patient_group['labels']

            current_size = dset_signals.shape[0]
            new_size = current_size + aug_signals.shape[0]

            dset_signals.resize((new_size, channels, time_steps))
            dset_labels.resize((new_size,))

            dset_signals[current_size:new_size, :, :] = aug_signals
            dset_labels[current_size:new_size] = aug_labels

            gc.collect()

    final_output_path = os.path.join(output_file_dir, f"{output_file_name}.h5")
    try:
        shutil.move(temp_output_file, final_output_path)
        print("Final .h5 file saved to:", final_output_path)
    except Exception as e:
        print("Failed to move .h5 file to output directory:", e)
