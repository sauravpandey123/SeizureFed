def get_stats_for_dataset(dataset_name, channel_name):
    name = dataset_name.lower() 
    if (name == 'chb'):
        mean, sd, index = chb_statistics(channel_name)
    elif (name == 'helsinki'):
        mean, sd, index = helsinki_statistics(channel_name)
    elif (name == 'nch'):
        mean, sd, index = nch_statistics(channel_name)
    elif (name == 'sienna'):
        mean, sd, index = sienna_statistics(channel_name)
    else:
        print ("bad name provided. options: {chb | helsinki | nch | sienna}")
    return mean, sd, index


def chb_statistics(channel_name):
    channel_means = [0.20765918, 0.20577365, 0.19865328, 0.2023162, 0.2027578, 0.20285356, 0.2019352, 0.20223266, 0.20304653, 0.20333311, 0.20370734, 0.20679545, 0.20545648, 0.20465624, 0.20595919, 0.20443487, 0.19996756, 0.19759804, 0.19681464]

    channel_sds =  [54.68643, 51.399017, 55.965202, 50.005814, 72.32885, 63.387005, 54.851936, 57.276337, 64.46384, 48.945732, 54.95025, 81.61089, 66.9985, 61.59204, 65.066284, 51.504505, 58.940712, 51.188946, 59.673237]

    channels = ['FP1-F7', 'F7-T7','T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
    
    channel_index = channels.index(channel_name)   
    return channel_means[channel_index], channel_sds[channel_index], channel_index
    
    
    
def helsinki_statistics(channel_name):
    channels = ['FP1-F7', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'FZ-CZ', 'CZ-PZ']

    channel_means =  [0.119284816, 0.8082072, 0.54967666, -0.12321772, -0.10221893, -0.016309947, -0.035373807, 0.0037955835, 0.40675756, -0.45838517, -0.3275061, -0.2186882]

    channel_sds =  [95.647675, 77.1941, 65.25772, 60.204952, 58.78162, 54.67052, 62.748707, 79.64921, 104.43702, 73.370995, 82.047195, 54.220375]

    channel_index = channels.index(channel_name)
    return channel_means[channel_index], channel_sds[channel_index], channel_index
    
    
    
def nch_statistics(channel_name):
    channels = ['C3-M2','O1-M2','O2-M1','CZ-O1','C4-M1','F4-M1','F3-M2','F3-C3','F4-C4']
    channel_means = [1.192093e-06, -3.33786e-06, -2.384186e-05, -3.325939e-05, -5.805492e-05, -5.960464e-05, -3.993511e-06, -4.947186e-06, -4.947186e-06]
    channel_sds = [0.0002441406, 0.0004882812, 0.0007324219, 0.0007324219, 0.0007719994, 0.0008096695, 0.0004882812, 0.0003452301, 0.0005979538]
    
    channel_index = channels.index(channel_name)   
    return channel_means[channel_index], channel_sds[channel_index], channel_index
    

def nch_min_max(idx):
    channels = ['C3-M2','O1-M2','O2-M1','CZ-O1','C4-M1','F4-M1','F3-M2','F3-C3','F4-C4']
    channel_mins = [-0.008712769, -0.008712769, -0.008712769, -0.008712769, -0.009590149, -0.01229858, -0.01130676, -0.008712769, -0.008728027]
    channel_maxs = [0.008712769, 0.008712769, 0.008712769, 0.008712769, 0.009979248, 0.01084137, 0.01222992, 0.008712769, 0.01102448]
    
    return channel_mins[idx], channel_maxs[idx]
    
    
    
    
def sienna_statistics(channel_name):
    channels = ['FP1', 'F3', 'FP2', 'F4','C3', 'C4', 'FP1-F3', 'FP2-F4', 'F3-C3', 'F4-C4']
    channel_means = [-8.713184, -8.824183, -7.121622, -8.161074, -11.204784, -15.494318, -20.479345, -26.834543, -34.949833, -40.107224]
    channel_sds = [97.37256, 90.50467, 76.20723, 85.0738, 100.63796, 125.65133, 222.17188, 229.2153, 279.97455, 335.1695]
    
    channel_index = channels.index(channel_name)
    return channel_means[channel_index], channel_sds[channel_index], channel_index


