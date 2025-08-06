import os
import torch
import numpy as np
import pandas as pd

from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchaudio

# find device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CreateDataset(Dataset):

    # initialize dataset variables
    def __init__(self, sound_folder, device):
        # variables
        self.TARGET_SAMPLE_RATE = 22050
        self.NUM_SAMPLES = 44100

        self.device = device
        self.folder = sound_folder
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.TARGET_SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)

        # count amount of files
        subfolders = [os.path.join(self.folder, i) for i in os.listdir(self.folder) if os.path.isdir(os.path.join(self.folder, i))]
        self.folderA = [os.path.join(subfolders[0],i) for i in os.listdir(subfolders[0]) if os.path.isfile(os.path.join(subfolders[0],i)) and os.path.splitext(os.path.join(subfolders[0],i))[1] == ".wav"]
        self.folderB = [os.path.join(subfolders[1],i) for i in os.listdir(subfolders[1]) if os.path.isfile(os.path.join(subfolders[1],i)) and os.path.splitext(os.path.join(subfolders[1],i))[1] == ".wav"]
        self.filenames = self.folderA + self.folderB
    
    # len func
    def __len__(self):
        return len(self.filenames)
    
    # get item
    def __getitem__(self, index):
        # get audio path
        audio_path = self.filenames[index]

        # get lable 0 = Fake 1 = Real
        # get worried about if this breaks w shuffle
        if audio_path in self.folderA: label = 0
        elif audio_path in self.folderB: label = 1

        signal, sr = torchaudio.load(audio_path)
        signal = self._resample_if_necessary(signal, sr)
        if signal.shape[0] > 1: signal = torch.mean(signal, dim=0, keepdim=True)
        if signal.shape[1] > self.NUM_SAMPLES: signal = signal[:, :self.NUM_SAMPLES]
        signal = self._right_pad_if_necessary(signal)
        signal = self.transform(signal)

        return signal, label
        # continue
    
    def _resample_if_necessary(self, signal, sr):
        if sr != self.TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.TARGET_SAMPLE_RATE)
            signal = resampler(signal)
        return signal
    
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.NUM_SAMPLES:
            num_missing_samples = self.NUM_SAMPLES - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

# define hyperparameters
batchsize = 100
sequence_len = 35 # chat
input_len = 128 # chat
hidden_size = 128
num_layers = 2
num_classes = 2
num_epochs = 5
learning_rate = 0.01

# training and validation data, no testing here
train_data = CreateDataset("data/training", device)
valid_data = CreateDataset("data/validation", device)

train_dataloader = DataLoader(train_data, batch_size=batchsize)
valid_dataloader = DataLoader(valid_data, batch_size=batchsize)

for batch in train_dataloader:
    x,y = batch
    print(x.shape)
    print(y.shape)
    break

#continue later
class CNNLSTM(nn.Module):
    def __init__(self, input_len, hidden_size, num_classes, num_layers):
        super(CNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)
    
    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)

        X = X.permute(0, 2, 3, 1)  # (B, 5, 7, 128) chat
        X = X.reshape(-1, sequence_len, input_len) # chat

        hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size) # state of the hidden layers (short term memory)
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size) # long term memory
        out, _ = self.lstm(X, (hidden_states, cell_states))
        out = self.output_layer(out[:, -1, :]) # flatten before output layer
        return out

model = CNNLSTM(input_len, hidden_size, num_classes, num_layers)
print(model)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

def train(num_epochs, model, train_dataloader, valid_dataloader, loss_function, optimizer):
    steps_per_epoch = len(train_dataloader)