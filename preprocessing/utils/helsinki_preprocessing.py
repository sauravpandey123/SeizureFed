import pandas as pd 
import numpy as np 
import os
import h5py
from scipy.signal import butter, filtfilt, iirnotch, sosfilt
from scipy.signal import decimate


#Function to create channels that match with CHB-MIT
def create_needed_channels(edf_reader, needed_channels, current_channels):
    current_channels = [channel.lower() for channel in current_channels]
    signal_cache = {}
    for index, channel in enumerate(current_channels):
        signal_cache[channel] = edf_reader.readSignal(index)
    all_new_signals = []
    channels_found_and_created = []
    for needed_channel in needed_channels:
        electrodes = needed_channel.split("-")
        first_electrode = f"eeg {electrodes[0].lower()}-ref"
        second_electrode = f"eeg {electrodes[1].lower()}-ref"
        signal_first_electrode = signal_cache.get(first_electrode)
        signal_second_electrode = signal_cache.get(second_electrode)
        if signal_first_electrode is not None and signal_second_electrode is not None:
            new_signal = signal_first_electrode - signal_second_electrode
            all_new_signals.append(new_signal)
            channels_found_and_created.append(needed_channel)
        else:
            missing = []
            if signal_first_electrode is None:
                missing.append(first_electrode)
            if signal_second_electrode is None:
                missing.append(second_electrode)
    return all_new_signals, channels_found_and_created


#Funciton to create the signals into patches
def create_patches_for_signals(y, patch_intervals, sampling_frequency = 256):
    total_samples_per_patch = sampling_frequency * patch_intervals 
    total_samples_existing = y.shape[-1] 
    total_patches = total_samples_existing // total_samples_per_patch 
    extra_samples = total_samples_existing % total_samples_per_patch 
    samples_to_consider = total_samples_existing - extra_samples
    data = y[:, :samples_to_consider]
    reshaped_data = data.reshape(total_patches, total_samples_per_patch, -1)
    final_data = reshaped_data.transpose(0, 2, 1)
    return final_data 


#Function to make sure the labels are now turned into patches
def  create_patches_for_labels(y, patch_intervals, sampling_frequency = 256):
    threshold = 0.5 #if more than 50% ones, then give the label 1 else 0 
    total_samples_per_patch = sampling_frequency * patch_intervals 
    total_samples_existing = y.shape[-1] 
    total_patches = total_samples_existing // total_samples_per_patch 
    extra_samples = total_samples_existing % total_samples_per_patch 
    samples_to_consider = total_samples_existing - extra_samples
    data = y[:samples_to_consider]
    reshaped_data = data.reshape(total_patches, total_samples_per_patch)
    refined_labels = (np.mean(reshaped_data, axis=1) > threshold).astype(int)
    return refined_labels


#Pass in an edf file name, obtain the labels for this file, given by all three experts, do the majority vote, and return the labels
def get_labels_per_second(file_path, annotation_files):
    all_annotation_files = [os.path.join(annotation_files, file) for file in os.listdir(annotation_files)]

    file_number = file_path.split("/")[-1].replace("eeg","").replace(".edf","")
    all_expert_labels = []
    for file in all_annotation_files: 
        labels = get_labels_for_file(file, file_number)
        all_expert_labels.append(labels)
    df = pd.concat(all_expert_labels, axis=1)
    df_cleaned = df.dropna()
    majority_vote = df_cleaned.mode(axis=1)[0]
    majority_vote_series = pd.Series(majority_vote, index=df_cleaned.index)
    return majority_vote_series.values


#For a given annotation file, return its labels for the given file
#This is called by the get_labels_per_second function
def get_labels_for_file(file, file_number):
    df = pd.read_csv(file)
    labels = df[file_number]
    return labels
                    

#This ensures that the number of labels matches total samples in the signals.
#For instance, (12, 1520896) (1520896,)
def match_labels_with_samples(new_signals, labels_per_second, sampling_frequency=256):
    total_samples = new_signals.shape[-1]
    new_labels = np.zeros(total_samples)  # Initialize an array for all samples
    num_seconds = len(labels_per_second)  # Total seconds of label data
    for i in range(num_seconds):
        lower_index = i * sampling_frequency
        upper_index = min((i + 1) * sampling_frequency, total_samples)  # Ensure no overflow
        new_labels[lower_index:upper_index] = labels_per_second[i]  # Assign the label
    return new_labels


#This function is for obtaining pre-ictal and inter-ictal labels from just seizure data
def get_seizure_prediction_data(signals, labels, patch_intervals, preictal_period, postictal_period = 5):
    shortened_labels = []  # Collect labels as a list
    shortened_data = []    # Collect signals as a list
    seizure_exists = True
    preictal_seconds = preictal_period * 60  # Convert preictal period to seconds
    postictal_seconds = postictal_period * 60

    while seizure_exists:
        seizure_exists = False
        for index, label in enumerate(labels):
            if label == 1:  # Found a seizure onset
                pre_ictal_start = max(0, int(index - preictal_seconds / patch_intervals))  # Ensure valid start
                ictal_beginning = index  # Seizure onset
                ictal_end = index
                # Locate the end of ictal phase
                while ictal_end < len(labels) and labels[ictal_end] == 1:
                    ictal_end += 1
                seizure_exists = True
                break

        if not seizure_exists:
            # Add all remaining data as interictal
            shortened_labels.extend([0] * len(labels))
            shortened_data.extend(signals)  # Append remaining data
            break

        # Add interictal data before preictal
        if pre_ictal_start > 0:
            shortened_labels.extend([0] * pre_ictal_start)
            shortened_data.extend(signals[:pre_ictal_start])

        # Add preictal data
        shortened_labels.extend([1] * (ictal_beginning - pre_ictal_start))
        shortened_data.extend(signals[pre_ictal_start:ictal_beginning])

        # Postictal exclusion
        index_after_ignoring_postictal = ictal_end + int(postictal_seconds/patch_intervals)
        if index_after_ignoring_postictal < len(labels):
            labels = labels[index_after_ignoring_postictal:]
            signals = signals[index_after_ignoring_postictal:]
        else:
            break
    return np.array(shortened_data), np.array(shortened_labels)


#A function to print how many continous blocks of 1s exist 
def count_contiguous_ones(labels):
    count = 0
    in_group = False  
    for label in labels:
        if label == 1 and not in_group:
            count += 1
            in_group = True
        elif label == 0:
            in_group = False
    return count


def convert_to_h5(final_signals, final_labels, hdf_dir, file_path, idx = "file"):
    edf_file = file_path.split("/")[-1].replace(".edf", "-") + str(idx) + ".h5"
    h5_file_path = os.path.join(hdf_dir, edf_file)

    with h5py.File(h5_file_path, 'w') as hf:
        hf.create_dataset('signals', data=final_signals, compression="gzip")

        if np.isscalar(final_labels):
            final_labels = np.array([final_labels])  # Wrap scalar into a 1-element array
        hf.create_dataset('labels', data=final_labels, compression="gzip")
        
    print(f"Saved file: {h5_file_path}")


def low_pass_filter(data, cutoff, fs, order=4):
    sos = butter(order, cutoff, fs=fs, btype='low', output='sos')  
    return sosfilt(sos, data, axis=-1)  

def downsample_signal(data, downsampling_factor):
    return data[:, :, ::downsampling_factor]


def preprocess_eeg(data, original_fs=256, downsampling_factor=2):
    data = low_pass_filter(data, cutoff=original_fs / (2 * downsampling_factor), fs=original_fs, order = 2)
    data = downsample_signal(data, downsampling_factor)
    return data



