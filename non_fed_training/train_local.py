import argparse
import h5py
import os
import logging
from utils.datasets_statistics import get_channel_index_for_dataset
from sklearn.utils.class_weight import compute_class_weight
from utils.utils import *
from utils.dataloader import get_dataloader, get_balanced_dataloader
from utils.model_training import *
from utils.tiny_sleep_net import TinySleepNet
import torch
import torch.nn as nn
import numpy as np



def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # prevent duplicate logs
    
    formatter = logging.Formatter('%(message)s')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    

def main(dataset_path, model_save_dir, model_save_name, dataset_index, nch_dataset_path):
    dataset_names = ['chb', 'helsinki', 'nch', 'sienna']
    assert 0 <= dataset_index <= 3, f"dataset_index must be between 0 and 3 (0=chb, 1=helsinki, 2=nch, 3=sienna)"
    logging.info(f"Selected dataset: {dataset_names[dataset_index]}")

    all_combined_accuracies = []
    n_epochs = 2
    patience = 25

    for seed in [42, 43, 44, 45, 46]:
        set_seed(seed)
        best_val_loss = float("inf")
        epochs_no_improve = 0

        channel_name = 'F3-C3'
        batch_size = 64
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(model_save_dir, exist_ok=True)
        save_name = f"{model_save_name}_{seed}.pth"

        datasets_to_merge = [('chb', True), ('helsinki', True), ('nch', False), ('sienna', True)]
        dataset_name, scale_to_nch = datasets_to_merge[dataset_index]

        logging.info(f"Dealing with: {dataset_name}")
        channel_index = get_channel_index_for_dataset(dataset_name, channel_name)
        logging.info(f"Loading {dataset_name} dataset and selecting {channel_name} (index: {channel_index})...")

        X_sub, y_sub = load_patientwise_file(dataset_path, channel_index, scale_to_nch, nch_dataset_path)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_train_val_test_split(X_sub, y_sub, random_state=seed)

        all_data = np.concatenate([X_train, X_val, X_test], axis=0)
        global_mean, global_sd = np.mean(all_data), np.std(all_data)

        X_train = standardize_data(X_train, global_mean, global_sd)
        X_val = standardize_data(X_val, global_mean, global_sd)
        X_test = standardize_data(X_test, global_mean, global_sd)

        unique_labels = np.unique(y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_labels, y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        train_dataloader = get_balanced_dataloader(X_train, y_train, [dataset_name]*len(X_train), batch_size, shuffle=True)
        val_dataloader = get_dataloader(X_val, y_val, batch_size, shuffle=False)
        test_dataloader = get_dataloader(X_test, y_test, batch_size, shuffle=False)

        sleep_model = TinySleepNet(num_classes=2, Fs=12, kernel_size=4).to(device)

        logging.info("Training from scratch...")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, sleep_model.parameters()), lr=4e-5, weight_decay=1e-6)

        for epoch in range(n_epochs):
            train_loss, train_acc = train(sleep_model, device, train_dataloader, loss_fn, optimizer)
            val_loss, val_acc = validate(sleep_model, device, val_dataloader, loss_fn)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(sleep_model.state_dict(), os.path.join(model_save_dir, save_name))
                logging.info("Best model saved...")
            else:
                epochs_no_improve += 1

            logging.info(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}")
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
                break

        test_loss, test_acc, _, _ = test(sleep_model, device, test_dataloader, loss_fn)
        logging.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        all_combined_accuracies.append(test_acc)

        logging.info(f"SEED = {seed} DONE\n===================")

    all_combined_accuracies = np.array(all_combined_accuracies)
    logging.info(f"All Accuracies: {all_combined_accuracies}")
    logging.info(f"Mean Accuracy: {np.mean(all_combined_accuracies):.2f}")
    logging.info(f"Std Dev: {np.std(all_combined_accuracies):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TinySleepNet on EEG dataset (CHB, Helsinki, NCH, Sienna)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--dataset_path', type=str, default='../DATA/helsinki_patientwise.h5',
                        help='Path to the .h5 dataset (e.g., ../DATA/helsinki_patientwise.h5)')
    parser.add_argument('--model_save_dir', type=str, default='saved_models_local',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--model_save_name', type=str, default='helsinki',
                        help='Base filename for saved models (seed will be appended)')
    parser.add_argument('--dataset_index', type=int, choices=[0, 1, 2, 3], default=1,
                        help='Which dataset to use for training: 0=chb, 1=helsinki, 2=nch, 3=sienna')
    parser.add_argument('--nch_dataset_path', type=str, default='../DATA/nch_patientwise.h5',
                        help='Path to NCH dataset for scaling')
    parser.add_argument('--log_file', type=str, default='train_log', help='File to save log output (model_save_name will be prepended)')

    args = parser.parse_args()

    log_file_path = os.path.join(f'{args.model_save_name}_{args.log_file}.txt')
    setup_logger(log_file_path)
    print(f"Logging results to: {log_file_path}")


    logging.info("\n===== Running with Arguments =====")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info("==================================\n")

    main(args.dataset_path, args.model_save_dir, args.model_save_name, args.dataset_index, args.nch_dataset_path)
