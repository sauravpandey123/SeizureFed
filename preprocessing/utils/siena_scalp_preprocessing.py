import pyedflib
import datetime
import numpy as np
import re
from scipy.signal import decimate, butter, filtfilt, iirnotch, sosfilt


def get_signals_from_edf(edf_file_path, needed_channels):
    f = pyedflib.EdfReader(edf_file_path)
    n_channels = f.signals_in_file
    signal_labels = f.getSignalLabels()
    signal_labels_lowered = [label.lower() for label in signal_labels]

    needed_indices = []
    for needed_channel in needed_channels:
        for idx, original_channel in enumerate(signal_labels_lowered):
            if needed_channel in original_channel:
                needed_indices.append(idx)
                break 

    needed_indices = np.array(needed_indices)
    needed_signals = [f.readSignal(i) for i in needed_indices]
    needed_signals = np.array(needed_signals)
    return needed_signals


def get_corresponding_annotation_file(sample_file, root_dir):
    annotation_file_index = sample_file.split("/")[-1].split("-")[0]
    annotation_file_name = "Seizures-list-" + annotation_file_index + ".txt"
    annotation_file_path = root_dir + annotation_file_index + "/" +  annotation_file_name #Get Seizures-list-PNXX.txt 
    file_name = sample_file.split("/")[-1]  #Get the exact file name, like PN00-1, PN00-2, etc
    return annotation_file_path, file_name 


def extract_seizure_data(annotation_file_path, target_file):
    seizure_data = {}
    with open(annotation_file_path, 'r') as file:
        lines = file.readlines()
        found_target = False
        for line in lines:
            if target_file.lower() in line.lower():
                found_target = True
                seizure_data["file"] = target_file
            if found_target:
                if "Registration start time" in line:
                    registration_start = re.search(r"\d{2}.\d{2}.\d{2}", line).group()
                    seizure_data["registration_start"] = registration_start
                elif "Registration end time" in line:
                    registration_end = re.search(r"\d{2}.\d{2}.\d{2}", line).group()
                    seizure_data["registration_end"] = registration_end
                elif "Seizure start time" in line:
                    seizure_start = re.search(r"\d{2}.\d{2}.\d{2}", line).group()
                    seizure_data.setdefault("seizures", []).append({"start": seizure_start})
                elif "Seizure end time" in line:
                    seizure_end = re.search(r"\d{2}.\d{2}.\d{2}", line).group()
                    if seizure_data.get("seizures"):
                        seizure_data["seizures"][-1]["end"] = seizure_end
                if "Seizure end time" in line and "seizures" in seizure_data:
                    break
    return seizure_data


def low_pass_filter(data, cutoff, fs, order=4):
    sos = butter(order, cutoff, fs=fs, btype='low', output='sos')  
    return sosfilt(sos, data, axis=-1)  



def downsample_signal(data, downsampling_factor):
    return data[:, ::downsampling_factor]


def get_downsampled_signals(data, original_fs, target_fs):
    downsampling_factor = int(original_fs/target_fs)
    data = low_pass_filter(data, cutoff=original_fs / (2 * downsampling_factor), fs=original_fs, order = 2)
    data = downsample_signal(data, downsampling_factor)
    return data


def convert_hourly_to_seconds(time_str):
    # Replace periods with colons if needed
    time_str = time_str.replace('.', ':')
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

def get_seizure_labels(signals_downsampled, seizure_info, target_sampling_frequency):
    registration_start = seizure_info['registration_start']
    seizure_start = seizure_info['seizures'][-1]['start']
    seizure_end = seizure_info['seizures'][-1]['end']

    registration_start_seconds = convert_hourly_to_seconds(registration_start)
    seizure_start_seconds = convert_hourly_to_seconds(seizure_start)
    seizure_end_seconds = convert_hourly_to_seconds(seizure_end)

    # Add 24 hours if the seizure happens after midnight (i.e., earlier in time than registration)
    if seizure_start_seconds < registration_start_seconds:
        seizure_start_seconds += 24 * 3600  # Add 24 hours (86400 seconds)
        seizure_end_seconds += 24 * 3600

    relative_seizure_start_seconds = seizure_start_seconds - registration_start_seconds
    relative_seizure_end_seconds = seizure_end_seconds - registration_start_seconds

    relative_preictal_start_seconds = relative_seizure_start_seconds - 3600  # preictal is 1 hour before

    seizure_start_index = int(relative_seizure_start_seconds * target_sampling_frequency)
    seizure_end_index = int(relative_seizure_end_seconds * target_sampling_frequency)

    seizure_start_index = min(seizure_start_index, signals_downsampled.shape[1])
    seizure_end_index = min(seizure_end_index, signals_downsampled.shape[1])

    num_samples = signals_downsampled.shape[-1]

    labels = np.zeros(num_samples, dtype=int)

    labels[seizure_start_index:seizure_end_index] = 1  # ictal

    return np.array(signals_downsampled), np.array(labels)


#This function converts the downsampled signals into patches. For labels, it makes sure it matches the size of the signals
#E.g. if signals has a total of 2000 samples, then labels should also have 2000 samples
#Signals are reshaped into patches so of size (patch_size, channels, samples_per_patch), Labels will be reshaped later!

def patchify_signals_and_adjust_labels(signals_downsampled, seizure_labels, patch_size, target_sampling_frequency):
    n_channels = signals_downsampled.shape[0]
    total_samples = signals_downsampled.shape[1]
    samples_per_patch = patch_size * target_sampling_frequency
    num_patches = signals_downsampled.shape[1] // samples_per_patch
    remaining_samples = total_samples % samples_per_patch 
    if remaining_samples > 0:
        signals_downsampled = signals_downsampled[:, :-remaining_samples]
        seizure_labels = seizure_labels[:-remaining_samples]

    signals_downsampled = signals_downsampled.reshape(n_channels, num_patches, samples_per_patch)
    signals_downsampled = signals_downsampled.transpose(1,0,2)
    
    return signals_downsampled, seizure_labels



#This function first converts the seizure labels into patches and based on the samples in the patch, assigns a label to the patch
def get_seizure_labels_per_patch(seizure_labels, num_patches, samples_per_patch, threshold_ratio=0.5):
    patchwise_labels = seizure_labels.reshape(num_patches, samples_per_patch)
    
    sum_of_ones = patchwise_labels.sum(axis=1)

    # Number of samples that must be 1 to exceed threshold
    threshold_count = threshold_ratio * samples_per_patch

    # If a patch has more than 'threshold_count' 1's, label patch as 1; else 0
    seizure_labels_per_patch = (sum_of_ones > threshold_count).astype(int)

    return seizure_labels_per_patch


def get_labels_and_data(signals, labels, patch_intervals, preictal_period, postictal_period = 5):
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
                # print(f"Found Seizure: Onset={index}, End={ictal_end}")
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