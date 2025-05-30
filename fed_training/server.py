import torch
import copy

class Server:
    def __init__(self, model, clients, device):
        self.global_model = model.to(device)
        self.clients = clients
        self.device = device
        self.global_count = 0
        
        self.aggregation_weights = None
        
        
    def compute_global_mean_prg(self, clients, segment_len):
        
        total_masked_sum = None
        total_samples = 0

        for client in clients:
            masked_sum, count = client.get_masked_sum_with_prg(len(clients))
            total_samples += count
            if total_masked_sum is None:
                total_masked_sum = masked_sum
            else:
                total_masked_sum += masked_sum

        global_mean = total_masked_sum / total_samples / segment_len
        
        self.global_count = total_samples
        
        return global_mean
    
    
    def compute_global_std_prg(self, clients, global_mean, segment_len):
        
        total_masked_ssd = torch.tensor(0.0)

        for client in clients:
            masked_ssd = client.get_masked_std_with_prg(len(clients), global_mean)
            total_masked_ssd += masked_ssd
        
        global_variance = total_masked_ssd / self.global_count / segment_len
        global_std = torch.sqrt(global_variance)
        
        return global_std
    
    
    def compute_fedavg_weights(self, clients):
        
        sample_counts = [client.num_samples() for client in clients]
        total_samples = sum(sample_counts)

        self.aggregation_weights = [count / total_samples for count in sample_counts]
        
        print("FedAvg Aggregation Weights:")
        for client, weight, count in zip(clients, self.aggregation_weights, sample_counts):
            print(f"  Client '{client.name}' (samples={count}): weight = {weight:.6f}")
        

    def aggregate_simple(self, client_weights):
        new_state = copy.deepcopy(self.global_model.state_dict())
        for key in new_state:
            new_state[key] = sum(w[key] for w in client_weights) / len(client_weights)
        self.global_model.load_state_dict(new_state)
        
        
    def aggregate(self, client_weights):
        new_state = copy.deepcopy(self.global_model.state_dict())
        for key in new_state.keys():
            new_state[key] = sum(self.aggregation_weights[i] * client_weights[i][key] for i in range(len(client_weights)))
        self.global_model.load_state_dict(new_state)
        
    
    def train_fedavg(self, num_rounds, lr, weight_decay, local_epochs, save_path=None, weighted=True, eval_interval=10):
        for rnd in range(num_rounds):    
            print(f"\n--- Round {rnd + 1} ---")
            
            weights = []
            
            for client in self.clients:
                updated_weights = client.local_train(self.global_model, lr, weight_decay, local_epochs)
                weights.append(updated_weights)
                
            if weighted:
                self.aggregate(weights)
            else:
                self.aggregate_simple(weights)
                
            
            if rnd % eval_interval == 0:
                self.validate_all()
                

        if save_path:
            torch.save(self.global_model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
    

    def train_subset(self, num_rounds, lr, weight_decay, local_epochs, subset_size, save_path=None, eval_interval=10):
        
        for rnd in range(num_rounds):
            
            print(f"\n--- Round {rnd + 1} ---")
            
            weights = []
            
            for client in self.clients:
                updated_weights = client.local_train_with_subset(self.global_model, subset_size, lr, weight_decay, local_epochs)
                weights.append(updated_weights)
                
            self.aggregate_simple(weights)
            
            if rnd % eval_interval == 0:
                self.validate_all()

        if save_path:
            torch.save(self.global_model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    def evaluate_all(self):
        print("\n--- Final Evaluation ---")
        for client in self.clients:
            loss, acc = client.evaluate(self.global_model)
            print(f"{client.name}: Loss = {loss:.4f}, Acc = {acc:.4f}")
            
            
    def validate_all(self):
        print("\n--- Validation Performance ---")
        for client in self.clients:
            loss, acc = client.validate(self.global_model)
            print(f"{client.name}: Loss = {loss:.4f}, Acc = {acc:.4f}")
