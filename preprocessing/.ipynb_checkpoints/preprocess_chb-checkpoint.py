import os
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from pyedflib import EdfReader
import h5py
from utils.chb_preprocessing import *  # Only if you actually need it
from utils.chb_create_patientwise_h5 import get_patientwise_file

def readEdfFile(pathToFile, channels):
    f = EdfReader(pathToFile)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    signal_labels.append('seizure')
    sigbufs = np.zeros((f.getNSamples()[0], n + 1))
    for i in np.arange(n):
        sigbufs[:, i] = f.readSignal(i)
    sigbufs[:, n] = 0.0  # seizure label channel
    df = pd.DataFrame(data=sigbufs, columns=signal_labels)
    df = df.loc[:, channels]
    df = df.loc[:, ~df.columns.duplicated()]
    f._close()
    return df.columns, df.values

def get_seizure_period(file_location):
    bytes_array = list(Path(file_location).read_bytes())
    return int(str(bin(bytes_array[38]))[2:] + str(bin(bytes_array[41]))[2:], 2), bytes_array[49]

def read_and_store_data(dataset_folder, output_dir_individual, sample_rate, channels):
    patients = sorted([d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d)) and d.startswith('chb')])
    print(f"Found patient folders: {patients}")
    print ("Beginning conversion for individual patients...")
    os.makedirs(output_dir_individual, exist_ok=True)

    for p in patients[:2]:
        patient_path = os.path.join(dataset_folder, p)
        edf_files = sorted([f for f in os.listdir(patient_path) if f.endswith('.edf')])
        seizure_files = sorted([f for f in os.listdir(patient_path) if f.endswith('seizures')])
        patient_signals = None

        for e in edf_files:
            edf_path = os.path.join(patient_path, e)
            try:
                columns, sigbufs = readEdfFile(edf_path, channels)
            except Exception as ex:
                print(f"Error processing {edf_path}: {ex}")
                continue

            num_channels = sigbufs.shape[-1]
            if seizure_files and seizure_files[0].startswith(e):
                seizure_path = os.path.join(patient_path, seizure_files[0])
                start, length = get_seizure_period(seizure_path)
                for i in range(start * sample_rate, (start + length) * sample_rate + 1):
                    if i < sigbufs.shape[0]:
                        sigbufs[i][num_channels - 1] = 1.0
                seizure_files.pop(0)

            if patient_signals is None:
                patient_signals = sigbufs
            else:
                patient_signals = np.vstack((patient_signals, sigbufs))

        if patient_signals is not None:
            output_path = os.path.join(output_dir_individual, f"{p}_data.h5")
            with h5py.File(output_path, 'w') as hf:
                hf.create_dataset('data', data=patient_signals, compression="gzip")
            print(f"Data for patient {p} saved to: {output_path}")
        else:
            print(f"No valid EDF files processed for patient {p}, skipping .h5 save.")


            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess CHB-MIT EEG EDF files into patchified HDF5 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  
    )
    
    parser.add_argument("--chb_mit_file_dir", type=str, required=True,
                        help="Path to the main CHB-MIT dataset directory (e.g., physionet.org/files/chbmit/1.0.0/).")
    parser.add_argument("--output_dir_individual", type=str,
                        help="Directory to save the output HDF5 files for individual patients. Will be used later to create one, unified .h5 file", default = 'chb_dataset_h5_PID')
    
    parser.add_argument("--output_file_dir", type=str,
                        help="Directory to save the final h5 file that will be used for training later.", default = '../DATA')
    
    parser.add_argument("--output_file_name", type=str,
                        help="Name of the final combined h5 file to be used for training later.", default = 'chb_patientwise')
    
    args = parser.parse_args()

    channels = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ', 'seizure'
    ]

    read_and_store_data(args.chb_mit_file_dir, args.output_dir_individual, sample_rate=256, channels=channels)
    get_patientwise_file(args.output_dir_individual, args.output_file_dir, args.output_file_name)
