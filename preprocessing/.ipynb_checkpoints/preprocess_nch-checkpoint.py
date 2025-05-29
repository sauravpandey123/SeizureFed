import os
import numpy as np
import pandas as pd
import h5py
import argparse
from utils.nch_preprocessing import *
from utils.nch_create_pairwise import get_patientwise_file
from utils.signal_augmentation import augment_seizure_cases


def read_and_store_data(path, output_dir):
        
    all_needed_files = [
        '8665_5530', '18472_14581', '11362_10972', '12637_8776',
        '9604_24547', '9700_12472', '6361_6205', '19150_10174',
        '4021_1885', '2263_13063', '9229_8746', '16654_19984',
        '8233_6904', '12967_9037'
    ]
    print ("Only the following files have some form of seizure. Converting these...\n")
    print (all_needed_files)
    
    os.makedirs(output_dir, exist_ok=True)

    term = 'seizure'
    ch_c3 = 0
    ch_c4 = 4
    ch_f4 = 5
    ch_f3 = 6

    patch_intervals = 2
    preictal_period = 60
    postictal_period = 10
    default_overlap_values = {0: 0, 1: 0.75}
    auto_balance = False

    for file in all_needed_files:
        try:
            signals, sleep_labels, channels = get_signals_and_stages(path, file)
        except Exception as e:
            print(f"Error in {file}: {e}")
            continue

        channels_array = np.array(channels)
        parts = file.split("_")
        patID = parts[0]
        studID = parts[1]

        signals = filterChannels(signals, channels_array)

        # Add bipolar channels
        f3_c3 = signals[:, ch_f3, :] - signals[:, ch_c3, :]
        f4_c4 = signals[:, ch_f4, :] - signals[:, ch_c4, :]
        signals = np.concatenate([signals, f3_c3[:, None, :], f4_c4[:, None, :]], axis=1)
        channels_array = np.append(channels_array, ['F3-C3', 'F4-C4'])

        file_name = f'{patID}_{studID}'
        sample_labels = getCorrespondingLabel(path, file_name, term)
        
        #Convert signals and labels into tensors
        signals = torch.tensor(signals, dtype=torch.float32)
        sample_labels = torch.tensor(sample_labels, dtype=torch.int64)

        # Patching
        signals_patches = create_patches_for_signals(signals, patch_intervals)
        label_patches = create_patches_for_labels(sample_labels, patch_intervals)

        final_signals, final_labels = get_seizure_prediction_data(
            signals_patches, label_patches, patch_intervals, preictal_period
        )

        final_signals = final_signals.cpu().numpy()
        final_labels = final_labels.cpu().numpy()

        signals, labels, _ = augment_seizure_cases(
            final_signals, final_labels, default_overlap_values, auto_balance
        )

        hdf5_filename = os.path.join(output_dir, f"{patID}_{studID}.hdf5")
        try:
            with h5py.File(hdf5_filename, 'w') as hdf5_file:
                hdf5_file.create_dataset('signals', data=signals)
                hdf5_file.create_dataset('labels', data=np.array(labels))
            print(f"Saved: {hdf5_filename}")
        except Exception as e:
            print(f"Error saving file {hdf5_filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess NCH EDF files into patchified HDF5 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--nch_edf_dir", type=str, required=True,
                        help="Path to the NCH dataset directory containing the EDFs with TSVs. ")
    parser.add_argument("--output_dir_individual", type=str, default='nch_dataset_h5_PID',
                        help="Directory to save individual patient HDF5 files.")
    parser.add_argument("--output_file_dir", type=str, default='../DATA',
                        help="Directory to save the combined HDF5 file.")
    parser.add_argument("--output_file_name", type=str, default='nch_patientwise',
                        help="Filename of the combined HDF5 file.")
    parser.add_argument("--batch_size", type=int, default=5000,
                        help="Number of examples to process at once when combining files.")

    args = parser.parse_args()

    read_and_store_data(args.nch_edf_dir, args.output_dir_individual)
    print("\nNow creating final patientwise .h5 file...")
    get_patientwise_file(args.output_dir_individual, args.output_file_dir,
                         args.output_file_name, args.batch_size)
