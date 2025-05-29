import os
import numpy as np
import pandas as pd
from math import ceil
from random import shuffle
from datetime import timezone
from dateutil import parser

import torch
import h5py

import mne
from mne.io import read_raw_edf
from scipy.signal import butter, sosfilt
from scipy import interpolate



filter_channels = [
 'EEG C3-M2',
 'EEG O1-M2',
 'EEG O2-M1',
 'EEG CZ-O1',
 'EEG C4-M1',
 'EEG F4-M1',
 'EEG F3-M2',
]  


def low_pass_filter(data, cutoff, fs, order=2):
    sos = butter(order, cutoff, fs=fs, btype='low', output='sos')
    return sosfilt(sos, data, axis=-1)


#Get signals and labels from EDF files
def get_signals_and_stages(edf_dir, name, verbose=False, downsample=True):
    raw = load_study(edf_dir, name, exclude=['Patient Event'])
    freq = int(raw.info['sfreq'])  # Original sampling rate
    channels = raw.ch_names
    n_samples = raw.n_times
    
    INTERVAL = 30 

    EVENT_DICT = {
        'Sleep stage W': 0,
        'Sleep stage N1': 1,
        'Sleep stage N2': 2,
        'Sleep stage N3': 3,
        'Sleep stage R': 4,
    }

    events, event_id = mne.events_from_annotations(raw, event_id=EVENT_DICT, verbose=verbose)
    labels = []
    data = []

    for event in events:
        label, onset = event[[2, 0]]
        indices = [onset, onset + INTERVAL * freq]

        if indices[1] <= n_samples:
            interval_data = raw.get_data(channels, start=indices[0], stop=indices[1])
            data.append(interval_data)
            labels.append(label)

    labels = np.array(labels)
    data = np.array(data)  
    if downsample:
        target_freq = 128

        if freq != target_freq:
            nyquist = target_freq / 2
            data = low_pass_filter(data, cutoff=nyquist, fs=freq, order=2)

            if freq % target_freq == 0:
                k = freq // target_freq
                data = data[:, :, ::k]  # Simple downsampling
            else:
                # Interpolation fallback if not integer multiple
                x = np.linspace(0, INTERVAL, num=INTERVAL * freq)
                new_x = np.linspace(0, INTERVAL, num=INTERVAL * target_freq)
                f = interpolate.interp1d(x, data, kind='linear', axis=-1, assume_sorted=True)
                data = f(new_x)
    return np.array(data), labels, channels


#Load the EDF file
def load_study(edf_dir, name, preload=False, exclude=[], verbose='CRITICAL'):
    path = os.path.join(edf_dir, name + '.edf')
    raw = read_raw_edf(input_fname=path, exclude=exclude, preload=preload,
                                verbose=verbose)
    annotation_path = os.path.join(edf_dir, name + '.tsv')
    df = pd.read_csv(annotation_path, sep='\t')
    annotations = mne.Annotations(df.onset, df.duration, df.description)
    raw.set_annotations(annotations)
    raw.rename_channels({name: name.upper() for name in raw.info['ch_names']})
    return raw


def filterChannels(signals, channel_names): 
    channel_names = list(channel_names)
    channel_indices = np.array([channel_names.index(ch) for ch in filter_channels])
    selected_data = signals[:, channel_indices, :]  
    return selected_data    


#### Get the labels for the tasks ####
def findLabel(description):
    EVENT_DICT = {
        'oxygen desaturation':1,
        'eeg arousal':1,
        'central apnea':1,
        'obstructive apnea':2,
        'mixed apnea':3,
        'obstructive hypopnea':1,
        'hypopnea':2,
        'seizure':1
        }
    if description in EVENT_DICT:
        return EVENT_DICT[description]    

    

def getCorrespondingLabel(directory, tsvfile, term):      
    edf_files = [f.replace(".edf","") for f in os.listdir(directory) if f.endswith('.edf')]
    tsv_file = os.path.join(directory, tsvfile + ".tsv")
    df = pd.read_csv(tsv_file,sep='\t')
    labelsList = []

    correspondingInterval = [] # the thirty second intervals corresponding to the label
    indices = []
    sleep_stages = ['Sleep stage W','Sleep stage N1','Sleep stage N2','Sleep stage N3','Sleep stage R']
    
    if term == "apnea": searchFor = ["central apnea", "mixed apnea", "obstructive apnea"]
    if term == "desat": searchFor = ["oxygen desaturation"]
    if term == "eeg": searchFor = ["eeg arousal"]
    if term == "hypop": searchFor = ["obstructive hypopnea", "hypopnea"]
    if term == "seizure": searchFor = ["seizure"]
    
    for i in range(len(df)):
        description = df.at[i, 'description'].strip()
        if description in sleep_stages:
            labelsList.append(i)
 
            
    acceptable_digits = [findLabel(item) for item in searchFor]
    for tracker in range(len(labelsList)):
        if (labelsList[tracker] in acceptable_digits): #skip those that have a 1 since you have already scanned them
            continue
        current_tracker = tracker 
        next_tracker = current_tracker + 1 
        current_number = labelsList[current_tracker]  #current corresponding index
        if (next_tracker == len(labelsList)):  #handle the last element
            next_number = current_number+1
        else:
            next_number = labelsList[next_tracker] #next corresponding index
        for i in range(current_number, next_number,1):  #for the first time, it would be between 12 and 17  (current, next + 1)
            description = df.at[i, 'description'].strip()  #check if this is oxygen desat
            if searchFor[0] in description.lower():  #we found oxygen deset
                searchLabel = findLabel(searchFor[0])
                labelsList[current_tracker] = searchLabel
                duration = df.at[i, 'duration']
                event_onset = df.at[i, 'onset']
                start_sleep_stage = df.at[current_number, 'onset']
                available = ceil(start_sleep_stage + 30 - event_onset) #some issues with 29.999995
                if (available < duration):
                    # print (duration, start_sleep_stage, available)
                    remaining = duration - available
                    slots = ceil(remaining/30)
                    for j in range(1, slots + 1):
                        if (current_tracker + j < len(labelsList)):
                            labelsList[current_tracker + j] = searchLabel
                        else:
                            print ("Overflow detected at the end in file:", tsv_file)
    for i in range(len(labelsList)):
        if (labelsList[i] not in acceptable_digits):
            labelsList[i] = 0

    return (labelsList)



#Funciton to create the signals into patches
def create_patches_for_signals(y, patch_intervals, sampling_frequency = 128):
    total_samples_per_patch = sampling_frequency * patch_intervals 
    examples, channels, samples = y.shape 
    y = y.reshape(channels, examples * samples)
    total_samples_existing = y.shape[-1]
    total_patches = total_samples_existing // total_samples_per_patch 
    extra_samples = total_samples_existing % total_samples_per_patch 
    samples_to_consider = total_samples_existing - extra_samples
    data = y[:, :samples_to_consider]
    reshaped_data = data.reshape(total_patches, -1, total_samples_per_patch)
    return reshaped_data


#Function to make sure the labels are now turned into patches
def  create_patches_for_labels(y, patch_intervals, sampling_frequency = 128):
    #y is the same as original labels
    original_patch_size = 30 #we know each patch is 30 seconds initially
    extra_patches = int(original_patch_size/patch_intervals)
    new_labels = y.repeat_interleave(extra_patches)
    return new_labels 

    
#For a given annotation file, return its labels for the given file
#This is called by the get_labels_per_second function
def get_labels_for_file(file, file_number):
    df = pd.read_csv(file)
    labels = df[file_number]
    return labels
                    

#This ensures that the number of labels matches total samples in the signals.
#For instance, (12, 1520896) (1520896,)
def match_labels_with_samples(new_signals, labels_per_second, sampling_frequency=128):
    total_samples = new_signals.shape[-1]
    new_labels = np.zeros(total_samples)  # Initialize an array for all samples
    num_seconds = len(labels_per_second)  # Total seconds of label data
    for i in range(num_seconds):
        lower_index = i * sampling_frequency
        upper_index = min((i + 1) * sampling_frequency, total_samples)  # Ensure no overflow
        new_labels[lower_index:upper_index] = labels_per_second[i]  # Assign the label
    return new_labels


#This function is for obtaining pre-ictal and inter-ictal labels from just seizure data
def get_seizure_prediction_data(signals, labels, patch_intervals, preictal_period, postictal_period=5):
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
                while ictal_end < len(labels) and labels[ictal_end] == 1:
                    ictal_end += 1
                seizure_exists = True
                break

        if not seizure_exists:
            shortened_labels.extend([0] * len(labels))
            shortened_data.extend(signals)  # Append remaining data
            break

        # Add interictal data before preictal
        if pre_ictal_start > 0:
            for signal in signals[:pre_ictal_start]:
                shortened_data.append(signal)  # Append signal individually
            shortened_labels.extend([0] * pre_ictal_start)

        for signal in signals[pre_ictal_start:ictal_beginning]:
            shortened_data.append(signal)  # Append signal individually
        shortened_labels.extend([1] * (ictal_beginning - pre_ictal_start))

        # Postictal exclusion
        index_after_ignoring_postictal = ictal_end + int(postictal_seconds / patch_intervals)
        if index_after_ignoring_postictal < len(labels):
            labels = labels[index_after_ignoring_postictal:]
            signals = signals[index_after_ignoring_postictal:, :, :]
        else:
            break

    shortened_data = torch.stack([data.clone() for data in shortened_data])
    shortened_labels = torch.tensor(shortened_labels)

    return shortened_data, shortened_labels



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