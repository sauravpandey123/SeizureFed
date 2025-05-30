import torch
import torch.nn as nn
import copy
from utils.train_utils import train, test, validate
from utils.federated_utils import*
from utils.data_utils import*
from datasets_statistics import *

class Client:
    def __init__(self, cid, name, channel_name, scale_to_nch, batch_size, device):
        
        self.id = cid  # integer index of the client
        self.name = name
        self.device = device
        self.batch_size = batch_size
        
        print(f"Initializing : {self.name}")
        file = f'../DATA/{self.name.lower()}_patientwise.h5' # Change depending on where your h5 files are
        _, _, channel_index = get_stats_for_dataset(self.name, channel_name)
        
        print(f"Loading {self.name} with channel {channel_name} (index: {channel_index})...")
        
        self.X, self.y = load_patientwise_file(file, channel_index, scale_to_nch)
        
        print(f"Shape of {self.name} data and labels: {self.X.shape}, {self.y.shape}")
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_weights = None
        self.loss_fn = None
        
        
    def get_masked_sum_with_prg(self, total_clients):
        
        sum_shape = self.X.sum(axis=None).shape
        sum_mask = torch.zeros(sum_shape)
        
        count_mask = torch.tensor(0.0)
        
        for j in range(total_clients):
            if j == self.id:
                continue
                
            # Generate sum mask
            seed_sum = generate_shared_seed(self.id, j)
            shared_sum_mask = prg(seed_sum, sum_shape)
            
            if self.id < j:
                sum_mask += shared_sum_mask
            else:
                sum_mask -= shared_sum_mask
                
            
            # Generate count mask using separate seed
            seed_count = generate_shared_seed(self.id, j + 1000)  # Use a separate namespace to avoid collisions
            shared_count_mask = prg(seed_count, ())

            if self.id < j:
                count_mask += shared_count_mask
            else:
                count_mask -= shared_count_mask
                
                
        
        local_sum = self.X.sum(axis=None)
        masked_sum = local_sum + sum_mask
        
        sample_count = torch.tensor(self.X.shape[0], dtype=torch.float32)
        masked_count = sample_count + count_mask
        
        return masked_sum, masked_count
    
    
    def get_masked_std_with_prg(self, total_clients, global_mean):

        diffs = torch.tensor(self.X, dtype=torch.float32) - global_mean
        squared_diffs = diffs.pow(2)
        
        local_ssd = squared_diffs.sum()

        ssd_mask = torch.tensor(0.0)

        for j in range(total_clients):
            if j == self.id:
                continue

            # Mask for sum of squared differences (ssd)
            seed_ssd = generate_shared_seed(self.id, j + 2000)  # Separate seed space
            shared_ssd_mask = 0 #prg(seed_ssd, ())
            ssd_mask += shared_ssd_mask if self.id < j else -shared_ssd_mask

        masked_ssd = local_ssd + ssd_mask

        return masked_ssd
    
    def normalize_data(self, global_mean, global_std):
        
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_train_val_test_split(self.X, self.y)
        
        X_train = standardize_data(X_train, global_mean, global_std)
        X_val = standardize_data(X_val, global_mean, global_std)
        X_test = standardize_data(X_test, global_mean, global_std)
        
        print(f"Post-standardization mean of {self.name} train: {np.mean(X_train):.4f}")
        print(f"Post-standardization  std of {self.name} train: {np.std(X_train):.4f}")
        
        unique_labels = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=y_train)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        self.train_loader = get_dataloader(X_train, y_train, self.batch_size, shuffle=True)
        self.val_loader = get_dataloader(X_val, y_val, self.batch_size, shuffle=False)
        self.test_loader = get_dataloader(X_test, y_test, self.batch_size, shuffle=False)
        
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        
    
    def num_samples(self):
        return self.X.shape[0]
        
    
    def get_random_subset_dataloader(self, M):

        full_dataset = self.train_loader.dataset
        dataset_length = len(full_dataset)

        # Randomly sample indices without replacement
        indices = np.random.choice(dataset_length, size=M, replace=False)

        # Create a subset dataset using these indices
        subset_dataset = Subset(full_dataset, indices)

        # Create a new DataLoader for the subset
        subset_dataloader = DataLoader(
            subset_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_loader.num_workers,
            pin_memory=self.train_loader.pin_memory
        )
        
        return subset_dataloader
    
    
    def local_train(self, global_model, lr, weight_decay, local_epochs):

        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(local_epochs):
            train(model, self.device, self.train_loader, self.loss_fn, optimizer)

        return model.state_dict()
        

    def local_train_with_subset(self, global_model, subset_size, lr, weight_decay, local_epochs):
        
        # Create subset
        subset_loader = self.get_random_subset_dataloader(subset_size)

        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(local_epochs):
            train(model, self.device, subset_loader, self.loss_fn, optimizer)

        return model.state_dict()
    
    
    def evaluate(self, model):
        return test(model, self.device, self.test_loader, self.loss_fn)
    
    def validate(self, model):
        return validate(model, self.device, self.val_loader, self.loss_fn)
