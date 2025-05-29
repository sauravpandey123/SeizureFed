import numpy as np
from scipy.signal import butter, sosfilt, decimate


def get_standardized_channel(data, channel_name, skip=False):
    channel_index = channels.index(channel_name) 

    mean = 0 if skip else channel_means[channel_index]
    sd = 1 if skip else channel_sds[channel_index]

    selected_data = data[:, channel_index:channel_index+1, :] 

    standardized_data = (selected_data - mean) / sd

    return standardized_data


def count_contiguous_ones(labels):
    count = 0
    in_group = False  
    
    for label in labels:
        if label == 1 and not in_group:
            # Start of a new group of 1s
            count += 1
            in_group = True
        elif label == 0:
            # End of the current group
            in_group = False

    return count


def get_labels_and_data(signals, labels, patch_intervals, preictal_period, postictal_period = 5):
    shortened_labels = []
    shortened_data = []   
    seizure_exists = True
    preictal_seconds = preictal_period * 60  
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




def low_pass_filter(data, cutoff, fs, order=4):
    sos = butter(order, cutoff, fs=fs, btype='low', output='sos')  
    return sosfilt(sos, data, axis=-1)  


def batch_process_downsampling(data, batch_size, cutoff, fs, order):
        
    num_batches = (data.shape[0] + batch_size - 1) // batch_size  # Calculate number of batches
    downsampled_batches = []

    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, data.shape[0])
        batch = data[start:end]

        filtered_batch = low_pass_filter(batch, cutoff=cutoff, fs=fs, order=order)
        downsampled_batch = filtered_batch[:, :, ::2]
        downsampled_batches.append(downsampled_batch)

    return np.concatenate(downsampled_batches, axis=0)


def get_filled_labels(final_data, labels):
    labels_copy = labels.copy()

    for i in range(final_data.shape[0]):
        last_channel = final_data[i, -1, :]
        if np.all(last_channel == 1):
            labels_copy[i] = 1
        elif np.all(last_channel == 0):
            labels_copy[i] = 0
        else:
            labels_copy[i] = np.max(last_channel)
    return labels_copy