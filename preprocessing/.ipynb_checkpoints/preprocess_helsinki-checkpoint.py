import os
import pyedflib
import numpy as np 
import pandas as pd
import argparse

from utils.helsinki_preprocessing import *
from utils.signal_augmentation import augment_seizure_cases
from utils.helsinki_create_patientwise import get_patientwise_file


def read_and_store_data(edf_file_dir, annotations_folder, output_dir_individual):
    
    needed_channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3' ,'F3-C3' ,'C3-P3' , 'P3-O1' , 'FP2-F4' , 'F4-C4', 'C4-P4' , 'P4-O2' , 'FP2-F8', 'F8-T8', 'T8-P8' , 'P8-O2' , 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8']

    patch_intervals = 2 #window size in seconds
    preictal_period = 60 #mins before seizure to set as preictal and mins after seizure to discard for post-ictal
    postictal_period = 10 #mins after seizure that you want to ignore
    original_frequency = 256 #original frequency of the signals
    downsampling_factor = 2 #downsample to 128 Hz
    
    default_overlaps = {0:0, 1:0.62}  #overlap values for augmentation 

    all_edf_files = [os.path.join(edf_file_dir, file) for file in os.listdir(edf_file_dir)]

    for index in range(len(all_edf_files)):
        file_path = all_edf_files[index]
        try:
            edf_reader = pyedflib.EdfReader(file_path)
        except Exception as e:
            print ("error reading file")
            print (e)
            continue
        n_channels = edf_reader.signals_in_file  # Number of channels
        channel_labels = edf_reader.getSignalLabels()  # Channel names
        sampling_rate = edf_reader.getSampleFrequency(0)  # Sampling rate (Hz) of the first channel
        duration = edf_reader.file_duration  # Duration of the recording in seconds

        n_samples = edf_reader.getNSamples()[0]
        all_signals = np.zeros((n_channels, n_samples))
        for i in range(n_channels):
            all_signals[i, :] = edf_reader.readSignal(i)

        all_unique_channel_collection = [] #store all channels that were created

        try:
            new_signals, new_channels = create_needed_channels(
                edf_reader, needed_channels=needed_channels, current_channels=channel_labels
            )
            if new_channels not in all_unique_channel_collection:
                all_unique_channel_collection.append(new_channels)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            edf_reader.close()

        new_signals = np.array(new_signals)

        labels_per_second = get_labels_per_second(file_path, annotations_folder)
        labels_expanded = match_labels_with_samples(new_signals, labels_per_second) 

        new_signals_patchified = create_patches_for_signals(new_signals, patch_intervals)
        new_labels_patchified = create_patches_for_labels(labels_expanded, patch_intervals)
        signals_downsampled = preprocess_eeg(
            new_signals_patchified, original_fs=original_frequency, downsampling_factor=downsampling_factor
        )
        final_signals, final_labels = get_seizure_prediction_data(
            signals_downsampled, new_labels_patchified, patch_intervals, preictal_period, postictal_period
        )
        final_signals, final_labels, overlaps = augment_seizure_cases(final_signals, final_labels, default_overlaps, auto_balance = False)
        convert_to_h5(final_signals, final_labels, args.output_dir_individual, file_path)
        print ("---------------------")
        
        
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Helsinki EEG EDF files into patchified HDF5 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  
    )
    
    parser.add_argument("--helsinki_edf_dir", type=str, required=True,
                        help="Path to the Helsinki dataset directory containing the EDFs (e.g., Helsinki/__codes__and__data__/edf).")
    
    parser.add_argument("--annotations_folder", type=str,
                        help="Path to the Helsinki dataset directory containing the annotations (e.g., Helsinki/__codes__and__data__/annotations)",
                           required = True)
    
    parser.add_argument("--output_dir_individual", type=str,
                        help="Directory to save the output HDF5 files for individual patients. Will be used later to create one, unified .h5 file", default = 'helsinki_dataset_h5_PID')
    
    parser.add_argument("--output_file_dir", type=str,
                        help="Directory to save the final h5 file that will be used for training later.", default = '../DATA')
    
    parser.add_argument("--output_file_name", type=str,
                        help="Name of the final combined h5 file to be used for training later.", default = 'helsinki_patientwise')
    
    parser.add_argument("--batch_size", type=int, default="5000",
                        help="How many examples to process at once when creating final patientwise file. Adjust based on memory")
    
    args = parser.parse_args()

    read_and_store_data(args.helsinki_edf_dir, args.annotations_folder, args.output_dir_individual)
    get_patientwise_file(args.output_dir_individual, args.output_file_dir, args.output_file_name, args.batch_size)