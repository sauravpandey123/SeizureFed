import torch
import numpy as np
import os
# import dataloader
# from dataloader import get_dataloader
from random import shuffle
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim

class TinySleepNet(nn.Module):
    def __init__(self, num_classes=2, Fs=8, kernel_size=8):
        super(TinySleepNet, self).__init__()

        # Representation Learning Part
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=Fs // 2, stride=Fs // 4)
        self.maxpool1 = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=kernel_size, stride=1)
        self.maxpool2 = nn.MaxPool1d(kernel_size=kernel_size // 2, stride=kernel_size // 2)

        # Sequence Learning Part
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)

        # Output layer
        self.fc = nn.Linear(128, num_classes)
        
    
    def forward(self, features, hc):
        # features shape: (batch_size, 1, 3840)

        features = self.conv1(features)
        features = self.maxpool1(features)

        features = self.conv2(features)
        features = self.conv3(features)
        features = self.conv4(features)
        features = self.maxpool2(features)

        # Preparing for LSTM: (batch_size, seq_len, input_size)
        features = features.permute(0, 2, 1)
        self.lstm.flatten_parameters()
        features, hc = self.lstm(features, hc)

        # Output layer using the last time step
        features = self.fc(features[:, -1, :])
        return features, hc