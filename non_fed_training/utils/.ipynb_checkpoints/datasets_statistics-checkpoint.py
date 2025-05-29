import h5py
import numpy as np

def get_channel_index_for_dataset(dataset_name, channel_name):
    name = dataset_name.lower() 
    if (name == 'chb'):
        index = chb_statistics(channel_name)
    elif (name == 'helsinki'):
        index = helsinki_statistics(channel_name)
    elif (name == 'nch'):
        index = nch_statistics(channel_name)
    elif (name == 'sienna'):
        index = sienna_statistics(channel_name)
    else:
        print ("bad name provided. options: {chb | helsinki | nch | sienna}")
    return index


def chb_statistics(channel_name):

    channels = ['FP1-F7', 'F7-T7','T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
    
    channel_index = channels.index(channel_name)   
    return channel_index
    
    
    
def helsinki_statistics(channel_name):
    channels = ['FP1-F7', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'FZ-CZ', 'CZ-PZ']

    channel_index = channels.index(channel_name)
    return channel_index
    
    
    
def nch_statistics(channel_name):
    channels = ['C3-M2','O1-M2','O2-M1','CZ-O1','C4-M1','F4-M1','F3-M2','F3-C3','F4-C4']

    channel_index = channels.index(channel_name)   
    return channel_index
    

    
def nch_min_max(idx, nch_path):    
    with h5py.File(nch_path, 'r') as f:
        all_keys = list(f.keys())
        all_data = [f[key]['signals'][:] for key in all_keys]

    data = np.concatenate(all_data, axis=0)
    channel_mins = data.min(axis=(0, 2))
    channel_maxs = data.max(axis=(0, 2))
    
    return channel_mins[idx], channel_maxs[idx]
    
    
    
    
def sienna_statistics(channel_name):
    channels = ['FP1', 'F3', 'FP2', 'F4','C3', 'C4', 'FP1-F3', 'FP2-F4', 'F3-C3', 'F4-C4']
    
    channel_index = channels.index(channel_name)
    return channel_index


