from collections import Counter
import numpy as np 

#This function groups consecutive indices of the same label into sublists.
def get_sub_indices(labels):
    all_group_indices = []
    this_group_indices = [] 
    previous = labels[0]
    for idx, label in enumerate(labels):
        if label == previous: 
            this_group_indices.append(idx)
        else: 
            all_group_indices.append(this_group_indices)
            this_group_indices = [idx] 
        previous = label 

    all_group_indices.append(this_group_indices)
    return all_group_indices

#Get required overlap for each class for complete balance
def calculate_overlap_fraction(current_patches, desired_patches, patch_length):
    total_signal_length = current_patches * patch_length
    overlap_fraction = 1 - ((total_signal_length - patch_length) / ((desired_patches - 1) * patch_length))
    return overlap_fraction



#Get the percentage of overlap required for each label to create a balanced class
#If we see extreme labels for any of the classes, then use the default overlap values
def get_overlap_percent_per_label(labels, n_examples, n_channels, patch_length, default_overlap_values):
    label_counts = Counter(labels)
    preictal_patches = label_counts.get(1, 0)
    interictal_patches = label_counts.get(0, 0)
    desired_patches = max(preictal_patches, interictal_patches)
    
    preictal_overlap = calculate_overlap_fraction(preictal_patches, desired_patches, patch_length)
    interictal_overlap = calculate_overlap_fraction(interictal_patches, desired_patches, patch_length)
    
    calculated_overlap_values = {0: interictal_overlap, 1: preictal_overlap}

    for key in calculated_overlap_values: 
        calculated_overlap = calculated_overlap_values[key]
        if calculated_overlap < 0 or calculated_overlap > 0.98:  #avoid errors or extremely high overlap
            calculated_overlap_values[key] = default_overlap_values[key]

    return calculated_overlap_values



def augment_seizure_cases(signals, labels, default_overlap_values, auto_balance = True):
    all_group_indices = get_sub_indices(labels)  #each sublist contains the indices of patches sharing the same label
    n_examples, n_channels, patch_length = signals.shape
    if (auto_balance == False):  #if no auto-balance, then just use default overlap values
        overlap_percent_per_label = default_overlap_values
    else: 
        overlap_percent_per_label = get_overlap_percent_per_label(labels, n_examples, n_channels, patch_length, default_overlap_values)

    signals_reshaped = signals.reshape(n_channels, -1)
    new_patches = []
    new_labels = []  
    for sub_group_indices in all_group_indices: 
        start_sample = sub_group_indices[0] * patch_length  #where do you begin augmenting
        end_sample = start_sample + len(sub_group_indices) * patch_length #where to end augmenting
        sub_group_label = labels[sub_group_indices[0]]  #get the label for this cluster
        this_label_overlap_percent = overlap_percent_per_label[int(sub_group_label)]  #find out the overlap size for this label
        step_size = int(patch_length * (1 - this_label_overlap_percent))  #get how much to skip by
        while start_sample + patch_length <= end_sample:  #do not overflow into the next patch
            selected_sub_signal = signals_reshaped[:, start_sample: start_sample + patch_length]
            start_sample = start_sample + step_size 
            new_patches.append(selected_sub_signal)
            new_labels.append(sub_group_label)

    return np.array(new_patches), np.array(new_labels), overlap_percent_per_label