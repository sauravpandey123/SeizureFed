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

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    
def main(chb, helsinki, nch, sienna, model_save_dir, model_save_name):
    logger = logging.getLogger()
    all_combined_accuracies = [] 
    n_epochs = 500
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
        dataset_paths = [chb, helsinki, nch, sienna]

        all_X_train, all_y_train = [], []
        all_X_val, all_y_val = [], []
        all_X_test, all_y_test = [], []
        train_names, val_names, test_names = [], [], []  
        all_data = [] 

        for idx, (dataset_name, scale_to_nch) in enumerate(datasets_to_merge):
            logger.info(f'Dealing with: {dataset_name}')
            file = dataset_paths[idx]
            channel_index = get_channel_index_for_dataset(dataset_name, channel_name)
            logger.info(f"Loading {dataset_name} dataset and selecting {channel_name} (index: {channel_index})...")

            X_sub, y_sub = load_patientwise_file(file, channel_index, scale_to_nch, nch)
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_train_val_test_split(X_sub, y_sub, random_state=seed)

            all_X_train.append(X_train); all_y_train.append(y_train)
            all_X_val.append(X_val); all_y_val.append(y_val)
            all_X_test.append(X_test); all_y_test.append(y_test)
            all_data.extend([X_train, X_val, X_test])
            train_names.extend([dataset_name] * len(X_train))
            val_names.extend([dataset_name] * len(X_val))
            test_names.extend([dataset_name] * len(X_test))

        X_train = np.concatenate(all_X_train); y_train = np.concatenate(all_y_train)
        X_val = np.concatenate(all_X_val); y_val = np.concatenate(all_y_val)
        X_test = np.concatenate(all_X_test); y_test = np.concatenate(all_y_test)
        all_data = np.concatenate(all_data)

        global_mean, global_sd = np.mean(all_data), np.std(all_data)
        X_train = standardize_data(X_train, global_mean, global_sd)
        X_val = standardize_data(X_val, global_mean, global_sd)
        X_test = standardize_data(X_test, global_mean, global_sd)

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        train_dataloader = get_balanced_dataloader(X_train, y_train, train_names, batch_size, shuffle=True)
        val_dataloader = get_dataloader(X_val, y_val, batch_size, shuffle=False)
        test_dataloader = get_dataloader(X_test, y_test, batch_size, shuffle=False)

        model = TinySleepNet(num_classes=2, Fs=12, kernel_size=4).to(device)
        logger.info("Training from scratch...")
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-5, weight_decay=1e-6)

        for epoch in range(n_epochs):
            train_loss, train_acc = train(model, device, train_dataloader, loss_fn, optimizer)
            val_loss, val_acc = validate(model, device, val_dataloader, loss_fn)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(model_save_dir, save_name))
                logger.info("Best model saved...")
            else:
                epochs_no_improve += 1

            logger.info(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}")
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
                break

        test_loss, test_acc, _, _ = test(model, device, test_dataloader, loss_fn)
        logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        all_combined_accuracies.append(test_acc)
        logger.info(f"SEED = {seed} DONE\n===================")

    logger.info("All Accuracies: %s" % all_combined_accuracies)
    logger.info("Mean Accuracy: %.4f" % np.mean(all_combined_accuracies))
    logger.info("Std Dev: %.4f" % np.std(all_combined_accuracies))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinySleepNet on combined datasets (CHB, Helsinki, NCH, Sienna)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--chb', type=str, default='../DATA/chb_patientwise.h5', help='Path to CHB dataset')
    parser.add_argument('--helsinki', type=str, default='../DATA/helsinki_patientwise.h5', help='Path to Helsinki dataset')
    parser.add_argument('--nch', type=str, default='../DATA/nch_patientwise.h5', help='Path to NCH dataset')
    parser.add_argument('--sienna', type=str, default='../DATA/siena_patientwise.h5', help='Path to Sienna dataset')
    parser.add_argument('--model_save_dir', type=str, default='saved_models_local', help='Directory to save models')
    parser.add_argument('--model_save_name', type=str, default='train_all', help='Base name for saved models')
    parser.add_argument('--log_file', type=str, default='training_all_log.txt', help='File to save log output')

    args = parser.parse_args()
    setup_logger(args.log_file)
    main(args.chb, args.helsinki, args.nch, args.sienna, args.model_save_dir, args.model_save_name)