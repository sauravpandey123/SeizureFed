# cross_dataset_eval.py
# 0 = chb, 1 = helsinki, 2 = nch, 3 = sienna

import argparse
import h5py
import os
import logging
from utils.datasets_statistics import get_channel_index_for_dataset
from sklearn.utils.class_weight import compute_class_weight
from utils.utils import *
from utils.dataloader import get_dataloader
from utils.model_training import *
from utils.tiny_sleep_net import TinySleepNet
import torch
import torch.nn as nn
import numpy as np

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

def main(chb, helsinki, nch, sienna, test_index, use_index, model_save_dir, model_save_name, log_path):
    setup_logger(log_path)

    all_test_accuracies = [] 
    batch_size = 64 

    datasets_to_merge = [('chb', True), ('helsinki', True), ('nch', False), ('sienna', True)]
    dataset_paths = [chb, helsinki, nch, sienna]

    TEST_ON = datasets_to_merge[test_index]
    USE = datasets_to_merge[use_index]
    test_dataset_name = TEST_ON[0]

    for seed in [42, 43, 44, 45, 46]:
        logging.info(f"SEED = {seed}")
        channel_name = 'F3-C3'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(seed)

        for idx, data in enumerate(datasets_to_merge):
            dataset_name = data[0]
            scale_to_nch = data[1]

            if dataset_name != test_dataset_name:
                logging.info(f"Skipping dataset: {dataset_name}")
                continue

            file = dataset_paths[idx]
            logging.info(f"Testing On: {dataset_name}")

            channel_index = get_channel_index_for_dataset(dataset_name, channel_name)
            X_sub, y_sub = load_patientwise_file(file, channel_index, scale_to_nch, nch)
            save_test_data, save_test_labels = X_sub, y_sub

        test_mean = np.mean(save_test_data)
        test_sd = np.std(save_test_data)

        standardized_test_data = standardize_data(save_test_data, test_mean, test_sd)
        test_dataloader = get_dataloader(standardized_test_data, save_test_labels, batch_size, shuffle=False)

        sleep_model = TinySleepNet(num_classes=2, Fs=12, kernel_size=4).to(device) 
        model_path = os.path.join(model_save_dir, f"{model_save_name}_{seed}.pth")
        sleep_model.load_state_dict(torch.load(model_path, weights_only = True))

        loss_fn = nn.CrossEntropyLoss()

        logging.info("Testing...")
        test_loss, test_acc, _, _ = test(sleep_model, device, test_dataloader, loss_fn)
        logging.info(f"Test Accuracy: {test_acc:.2f}%")
        all_test_accuracies.append(test_acc)
        logging.info("******************")

    logging.info("*** FINAL RESULTS ***")
    logging.info(f"Model Trained On: {USE[0]} | Evaluated On: {TEST_ON[0]}")
    all_test_accuracies = np.array(all_test_accuracies)
    logging.info(f"Mean Accuracy: {np.mean(all_test_accuracies):.2f}%")
    logging.info(f"Std Deviation: {np.std(all_test_accuracies):.2f}%")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TinySleepNet on a specified dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--chb', type=str, default='../DATA/chb_patientwise.h5',
                        help='Path to CHB dataset')
    parser.add_argument('--helsinki', type=str, default='../DATA/helsinki_patientwise.h5',
                        help='Path to Helsinki dataset')
    parser.add_argument('--nch', type=str, default='../DATA/nch_patientwise.h5',
                        help='Path to NCH dataset')
    parser.add_argument('--sienna', type=str, default='../DATA/siena_patientwise.h5',
                        help='Path to Sienna dataset')

    parser.add_argument('--test_index', type=int, choices=[0,1,2,3], default=2,
                        help='Index of dataset to test on: 0 = chb, 1 = helsinki, 2 = nch, 3 = sienna')
    parser.add_argument('--use_index', type=int, choices=[0,1,2,3], default=1,
                        help='Index of dataset the model was trained on: 0 = chb, 1 = helsinki, 2 = nch, 3 = sienna')

    parser.add_argument('--model_save_dir', type=str, default='saved_models_local',
                        help='Directory where model checkpoints are saved')
    parser.add_argument('--model_save_name', type=str, default='helsinki',
                        help='Base filename for the individual saved models (seeds appended)')
    parser.add_argument('--log_path', type=str, default='cross_eval_log.txt',
                        help='Path to save the evaluation log')

    args = parser.parse_args()

    logging.info("\n===== Running with Arguments =====")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info("==================================\n")

    main(args.chb, args.helsinki, args.nch, args.sienna, args.test_index,
         args.use_index, args.model_save_dir, args.model_save_name, args.log_path)
