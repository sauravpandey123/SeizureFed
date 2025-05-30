import argparse
import os
import torch
import numpy as np

from utils.train_utils import set_seed
from models.tiny_sleep_net import TinySleepNet
from client import Client
from server import Server

def main(args):
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.model_save_dir, exist_ok=True)
    
    datasets = [('chb', True), ('helsinki', True), ('nch', False), ('sienna', True)]
    
    clients = [Client(i, datasets[i][0], args.channel_name, datasets[i][1], args.batch_size, device) for i in range(len(datasets))]
        
    # Initialize global model and server
    global_model = TinySleepNet(num_classes=2, Fs=12, kernel_size=4)
    server = Server(model=global_model, clients=clients, device=device, )
    
    global_mean = server.compute_global_mean_prg(clients, args.segment_len)
    global_std = server.compute_global_std_prg(clients, global_mean, args.segment_len)
    
    print("Global Mean of all examples :", f"{global_mean.item():.6}")
    print("Global STD  of all examples :", f"{global_std.item():.6}")  
    
    for client in clients:
        client.normalize_data(global_mean.numpy(), global_std.numpy())

    # Train
    model_save_path = os.path.join(
        args.model_save_dir,
        f"{args.model_prefix}_subset{args.subset_size}_rounds{args.num_rounds}.pth"
    )
    
    if args.method == "rsa":
        print("Training with Random Subset Aggregation with M = %s" % args.subset_size)
    
        server.train_subset(
            num_rounds=args.num_rounds,
            lr=args.lr,
            weight_decay=args.weight_decay,
            local_epochs=args.local_epochs,
            subset_size=args.subset_size,
            save_path=model_save_path
        )
        
    elif args.method == "fedavg_simple":
        print("Training with simple FedAvg (unweighted)")
    
        server.train_fedavg(
            num_rounds=args.num_rounds,
            lr=args.lr,
            weight_decay=args.weight_decay,
            local_epochs=args.local_epochs,
            save_path=model_save_path,
            weighted=False
        )
        
    elif args.method == "fedavg_weighted":
        print("Training with FedAvg (weighted)")
        
        server.compute_fedavg_weights(clients)
    
        server.train_fedavg(
            num_rounds=args.num_rounds,
            lr=args.lr,
            weight_decay=args.weight_decay,
            local_epochs=args.local_epochs,
            save_path=model_save_path,
            weighted=True
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # Final evaluation
    server.evaluate_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Training with Client-Server Simulation")
    parser.add_argument("--subset_size", type=int, default=10000, help="Subset size M for each client")
    parser.add_argument("--model_save_dir", type=str, default="./saved_models", help="Directory to save model checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use")
    parser.add_argument("--model_prefix", type=str, default="fed_model", help="Prefix for saved model filename")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of federated training rounds")
    parser.add_argument("--eval_interval", type=int, default=100, help="Number of rounds between global model validation")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--local_epochs", type=int, default=1, help="Number of local epochs per client")
    parser.add_argument("--channel_name", type=str, default="F3-C3", help="EEG channel to use")
    parser.add_argument("--segment_len", type=int, default=256, help="EEG segment length")
    parser.add_argument("--method", type=str, default="rsa", help="Aggregation method: rsa (random subset aggregation), fedavg_simple, or fedavg_weighted")
    

    args = parser.parse_args()
    main(args)
