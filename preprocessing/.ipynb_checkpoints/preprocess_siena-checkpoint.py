import os
import argparse
import torch
import numpy as np
import h5py
from random import shuffle
from utils.siena_scalp_preprocessing import *
from utils.signal_augmentation import *
from utils.siena_scalp_create_patientwise_h5 import create_patientwise_h5

def main(args):
    os.makedirs(args.output_dir_individual, exist_ok=True)

    edf_file_paths = []
    for dirpath, dirnames, filenames in os.walk(args.dataset_dir):
        for filename in filenames:
            if filename.lower().endswith('.edf'):
                file_path = os.path.join(dirpath, filename)
                edf_file_paths.append(file_path)

    print(f"Found {len(edf_file_paths)} .edf files.")
    shuffle(edf_file_paths)

    label_count = {0: 0, 1: 0}

    for edf_file_path in edf_file_paths:
        print("===================")

        needed_channels = ['fp1', 'f3', 'fp2', 'f4', 'c3', 'c4']
        signals = get_signals_from_edf(edf_file_path, needed_channels)
        annotation_file_path, file_name = get_corresponding_annotation_file(edf_file_path, args.dataset_dir)

        signals_downsampled = get_downsampled_signals(signals, args.current_fs, args.target_fs)
        seizure_info = extract_seizure_data(annotation_file_path, file_name)

        signals_downsampled, seizure_labels = get_seizure_labels(
            signals_downsampled, seizure_info, args.target_fs
        )

        # Bipolar combinations
        fp1_f3 = (signals_downsampled[0, :] - signals_downsampled[1, :])[np.newaxis, :]
        fp2_f4 = (signals_downsampled[2, :] - signals_downsampled[3, :])[np.newaxis, :]
        f3_c3 = (signals_downsampled[1, :] - signals_downsampled[4, :])[np.newaxis, :]
        f4_c4 = (signals_downsampled[3, :] - signals_downsampled[5, :])[np.newaxis, :]
        signals_downsampled = np.concatenate([signals_downsampled, fp1_f3, fp2_f4, f3_c3, f4_c4], axis=0)

        samples_per_patch = args.patch_size * args.target_fs
        signals_patchified, labels_trimmed = patchify_signals_and_adjust_labels(
            signals_downsampled, seizure_labels, args.patch_size, args.target_fs
        )

        num_patches = signals_patchified.shape[0]
        labels_patchified = get_seizure_labels_per_patch(
            labels_trimmed, num_patches, samples_per_patch
        )

        new_signals, new_labels = get_labels_and_data(
            signals_patchified, labels_patchified, args.patch_size, args.preictal, args.postictal
        )

        final_signals_patchified, final_labels_patchified, _ = augment_seizure_cases(
            new_signals, new_labels, {0: 0, 1: args.seizure_overlap}, args.auto_balance
        )

        unique_labels, counts = np.unique(final_labels_patchified, return_counts=True)
        for label, count in zip(unique_labels, counts):
            label_count[label] += count

        pat_id = edf_file_path.split("/")[-1].split(".")[0]
        example_filename = os.path.join(args.output_dir_individual, f"{pat_id}.h5")

        with h5py.File(example_filename, 'w') as f:
            f.create_dataset('signals', data=final_signals_patchified)
            f.create_dataset('labels', data=final_labels_patchified)

        print(f"Saved file: {example_filename}")

    print ("Now creating final patientwise .h5 file...")
    
    create_patientwise_h5(args.output_dir_individual,  args.output_file_name, args.output_file_dir, args.batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Siena Scalp EEG EDF files into patchified HDF5 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  
    )
    
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to the main dataset directory containing the patient folders (PNXX) that have the EDF files.")
    
    parser.add_argument("--current_fs", type=int, default=512,
                        help="Sampling frequency of the original EEG data (Hz).")
    parser.add_argument("--target_fs", type=int, default=128,
                        help="Target downsampled frequency (Hz).")
    parser.add_argument("--patch_size", type=int, default=2,
                        help="Duration (in seconds) of each EEG patch.")
    parser.add_argument("--preictal", type=int, default=60,
                        help="Number of seconds before seizure onset to label as preictal.")
    parser.add_argument("--postictal", type=int, default=10,
                        help="Number of seconds after seizure end to include.")
    parser.add_argument("--seizure_overlap", type=float, default=0.4,
                        help="Overlap ratio for seizure samples during augmentation.")
    parser.add_argument("--auto_balance", action="store_true",
                        help="Enable automatic class balancing during augmentation.")

    parser.add_argument("--output_dir_individual", type=str, default="siena_dataset_h5_PID",
                        help="Directory to save the output HDF5 files for individual patients. Will be used later to create one, unified .h5 file")
    parser.add_argument("--output_file_name", type=str, default="siena_patientwise.h5",
                        help="Name of the final combined h5 file to be used for training later.")
    parser.add_argument("--output_file_dir", type=str, default="../DATA",
                        help="Directory to save the final h5 file that will be used for training later.")
    parser.add_argument("--batch_size", type=int, default="5000",
                        help="How many examples to process at once when creating final patientwise file. Adjust based on memory")


    args = parser.parse_args()
    main(args)

