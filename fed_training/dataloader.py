import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np 


class LazyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx], dtype=torch.long)
    
    

def get_dataloader(X, y, batch_size, shuffle, num_workers = 0, pin_memory = True):
    dataset = LazyDataset(X,y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle, num_workers=0, pin_memory=True)
    return dataloader


class LazyTrainDataset(Dataset):
    def __init__(self, X, y, train_names):
        self.X = X
        self.y = y
        self.train_names = np.array(train_names)  # Ensure it's a NumPy array
        self.indices_dict = {} 
        
        datasets = Counter(train_names)
        self.total_datasets = len(datasets)
        
        for key in datasets:
            self.indices_dict[key] = np.where(self.train_names == key)[0] 

        self.all_keys = list(datasets.keys())
        self.min_value = min(datasets.values())
        self.max_value = max(datasets.values())

    def __len__(self):
        return self.max_value * self.total_datasets 

    def __getitem__(self, idx):
        dataset_idx = idx % self.total_datasets
        key_to_select = self.all_keys[dataset_idx]

        indices = self.indices_dict[key_to_select]
        selected_index = np.random.choice(indices)  
            
        return torch.from_numpy(self.X[selected_index]).float(), torch.tensor(self.y[selected_index], dtype=torch.long)
    
    

def get_balanced_dataloader(X, y, train_names, batch_size, shuffle, num_workers = 0, pin_memory = True):
    dataset = LazyTrainDataset(X,y,train_names)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle, num_workers=0, pin_memory=True)
    return dataloader