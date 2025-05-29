import argparse
import os
import logging
import torch
import torch.nn as nn
import numpy as np

from utils.datasets_statistics import get_channel_index_for_dataset
from utils.utils import *
from utils.dataloader import get_dataloader
from utils.model_training import test
from utils.tiny_sleep_net import TinySleepNet


def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(message)s')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def main(chb, helsinki, nch, sienna, test_index, model_save_dir, model_save_name, log_path):
    setup_logger(log_path)

    batch_size = 64
    channel_name = 'F3-C3'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets_to_merge = [('chb', True), ('helsinki', True), ('nch', False), ('sienna', True)]
    dataset_paths = [chb, helsinki, nch, sienna]
    test_dataset_name, test_scale = datasets_to_merge[test_index]

    all_combined_accuracies, all_combined_f1s, all_combined_aurocs = [], [], []

    for seed in [42, 43, 44, 45, 46]:
        logging.info(f"\n=== SEED: {seed} ===")
        set_seed(seed)

        all_X_train, all_X_test, all_y_test, all_examples = [], [], [], []
        test_names = []

        for idx, (dataset_name, scale_to_nch) in enumerate(datasets_to_merge):
            file = dataset_paths[idx]
            channel_index = get_channel_index_for_dataset(dataset_name, channel_name)

            X_sub, y_sub = load_patientwise_file(file, channel_index, scale_to_nch, nch)
            (X_train, _), (X_val, _), (X_test, y_test) = stratified_train_val_test_split(X_sub, y_sub, random_state=seed)

            if dataset_name == test_dataset_name:
                save_test_data = X_test
                save_test_labels = y_test

            all_X_train.append(X_train)
            all_X_test.append(X_test)
            all_y_test.append(y_test)
            all_examples.extend([X_train, X_val, X_test])
            test_names.extend([dataset_name] * len(X_test))

        all_examples = np.concatenate(all_examples, axis=0)
        global_mean, global_sd = np.mean(all_examples), np.std(all_examples)

        logging.info(f"Standardizing dataset {test_dataset_name}")
        logging.info(f"Global mean: {global_mean:.4f}, Global std: {global_sd:.4f}")

        standardized_test_data = (save_test_data - global_mean) / global_sd
        test_dataloader = get_dataloader(standardized_test_data, save_test_labels, batch_size, shuffle=False)

        sleep_model = TinySleepNet(num_classes=2, Fs=12, kernel_size=4).to(device)
        model_path = os.path.join(model_save_dir, f"{model_save_name}_{seed}.pth")
        sleep_model.load_state_dict(torch.load(model_path, weights_only=True, map_location = torch.device('cpu')))

        loss = nn.CrossEntropyLoss()
        logging.info("Testing model...")
        test_loss, test_acc, f1, auroc = test(sleep_model, device, test_dataloader, loss)

        logging.info(f"Test Accuracy: {test_acc:.2f} | F1 Score: {f1:.2f} | AUROC: {auroc:.2f}")
        all_combined_accuracies.append(test_acc)
        all_combined_f1s.append(f1)
        all_combined_aurocs.append(auroc)

    logging.info("\n********** FINAL RESULTS ************")
    logging.info(f"Evaluated on: {test_dataset_name}")
    logging.info(f"Accuracy Mean: {np.mean(all_combined_accuracies):.2f} | Std: {np.std(all_combined_accuracies):.2f}")
    logging.info(f"F1 Score Mean: {np.mean(all_combined_f1s):.2f} | Std: {np.std(all_combined_f1s):.2f}")
    logging.info(f"AUROC Mean: {np.mean(all_combined_aurocs):.2f} | Std: {np.std(all_combined_aurocs):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate TinySleepNet Centralized models on different EEG datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--chb', type=str, default='../DATA/chb_patientwise.h5', help='Path to CHB dataset')
    parser.add_argument('--helsinki', type=str, default='../DATA/helsinki_patientwise.h5', help='Path to Helsinki dataset')
    parser.add_argument('--nch', type=str, default='../DATA/nch_patientwise.h5', help='Path to NCH dataset')
    parser.add_argument('--sienna', type=str, default='../DATA/siena_patientwise.h5', help='Path to Sienna dataset')
    parser.add_argument('--test_index', type=int, choices=[0, 1, 2, 3], default=2,
                        help='Index of dataset to test on (0 = chb, 1 = helsinki, 2 = nch, 3 = sienna)')
    parser.add_argument('--model_save_dir', type=str, default='saved_models_local', help='Directory with saved models')
    parser.add_argument('--model_save_name', type=str, default='train_all', help='Base name for saved models (seed will be appended)')
    parser.add_argument('--log_path', type=str, default='cross_eval_log.txt', help='Path to log file')

    args = parser.parse_args()

    main(args.chb, args.helsinki, args.nch, args.sienna, args.test_index,
         args.model_save_dir, args.model_save_name, args.log_path)
