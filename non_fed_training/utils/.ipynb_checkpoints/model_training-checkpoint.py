import torch
from torch.utils.data import TensorDataset, DataLoader
import h5py
import glob
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix
import os

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
import torch.nn.functional as F



def train(model, device, train_loader, criterion, optimizer):
    model.train() 
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  
        
        output, _ = model(data, None)
        
        loss = criterion(output, target)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, device, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output, _ = model(data, None)
            loss = criterion(output, target)
            running_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy





def test(model, device, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output, _ = model(data, None)
            loss = criterion(output, target)
            running_loss += loss.item()
            
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_logits.extend(output.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    
    test_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total

    # F1 Scores
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_binary = f1_score(all_labels, all_preds, average='binary')

    all_logits = torch.tensor(np.array(all_logits))
    all_probs = F.softmax(all_logits, dim=1).numpy()
    if all_probs.shape[1] == 2:
        auroc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auroc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

    return test_loss, accuracy, f1_binary, auroc





def load_saved_model(prior_dataset, channel):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_models = os.listdir('../models')  # List all available models

    if prior_dataset.lower() == "sienna":
        splits = channel.split("-")
        first, second = splits[0], splits[1]
        needed_model = f'sienna_trained_{first}_{second}.pth'

        if needed_model in all_models:
            model_path = f'../models/{needed_model}'
            print(f"Found model: {needed_model}, loading onto {device}...")

            model = torch.load(model_path, map_location=device) 
            model.to(device)
            return model, needed_model
        else:
            raise FileNotFoundError(f"Error: Model '{needed_model}' not found in '../models'. Available models: {all_models}")

    else:
        raise ValueError(f"Error: Unknown dataset '{prior_dataset}'.")


        