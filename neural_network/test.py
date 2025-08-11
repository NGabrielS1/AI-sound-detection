import os
import time
import torch
import random
import matplotlib.pyplot as plt
from statistics import mean, median

from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torchaudio

# find device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
torch.manual_seed(41)
random.seed(41)

#dataset class
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
        folderA = [os.path.join(subfolders[0],i) for i in os.listdir(subfolders[0]) if os.path.isfile(os.path.join(subfolders[0],i)) and os.path.splitext(os.path.join(subfolders[0],i))[1] == ".wav"]
        folderB = [os.path.join(subfolders[1],i) for i in os.listdir(subfolders[1]) if os.path.isfile(os.path.join(subfolders[1],i)) and os.path.splitext(os.path.join(subfolders[1],i))[1] == ".wav"]
        self.filenames = folderA + folderB
    
    # get length of data
    def __len__(self):
        return len(self.filenames)
    
    # get item
    def __getitem__(self, index):
        # get audio path
        audio_path = self.filenames[index]

        # get lable 0 = Fake 1 = Real
        # get worried about if this breaks w shuffle
        if os.path.basename(os.path.dirname(audio_path)) == "fake": label = 0
        elif os.path.basename(os.path.dirname(audio_path)) == "real": label = 1

        signal, sr = torchaudio.load(audio_path) #get audio signal & sample rate
        signal = self._resample_if_necessary(signal, sr)
        if signal.shape[0] > 1: signal = torch.mean(signal, dim=0, keepdim=True) #change channels
        if signal.shape[1] > self.NUM_SAMPLES: signal = signal[:, :self.NUM_SAMPLES] #cut signal
        signal = self._right_pad_if_necessary(signal) #pad signal
        signal = self.transform(signal)

        return signal, label
    
    #resample
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
batchsize = 1
sequence_len = 35
input_len = 128
hidden_size = 128
num_layers = 2
num_classes = 2

# Creating datasets and dataloaders
dataset = CreateDataset("data", device)
train_data, valid_data, test_data = random_split(dataset, [14296, 1787, 1787])
test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False)

#CNN-LSTM model
class CNNLSTM(nn.Module):
    def __init__(self, input_len, hidden_size, num_classes, num_layers):
        super(CNNLSTM, self).__init__()
        #variables
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        #CNN
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
        #LSTM
        self.lstm = nn.LSTM(input_len, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, num_classes)
    
    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)

        X = X.permute(0, 2, 3, 1)
        # X = X.reshape(-1, sequence_len, input_len)
        B, H, W, C = X.shape
        X = X.reshape(B, H * W, C)

        hidden_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size, device=X.device) # state of the hidden layers (short term memory)
        cell_states = torch.zeros(self.num_layers, X.size(0), self.hidden_size, device=X.device) # long term memory
        out, _ = self.lstm(X, (hidden_states, cell_states))
        out = self.output_layer(out[:, -1, :]) # flatten before output layer
        return out

#create model and load weights
model = CNNLSTM(input_len, hidden_size, num_classes, num_layers).to(device)
model.load_state_dict(torch.load(("neural_network/CNNLSTM_VOICE_NN.pt"),map_location=torch.device(device)))

#test function
def test(model, test_dataloader):
    y_true = []
    y_pred = []
    softmax = nn.Softmax(dim=1)
    model.eval()
    acc = 0
    with torch.no_grad():
        for batch, (image, label) in enumerate(test_dataloader):
            image, label = image.to(device), label.to(device)
            x = model(image)
            x = softmax(x)
            pred = torch.argmax(x, dim=1)
            y_true.append(label.item())
            y_pred.append(pred.item())
    
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Fake", "Real"]).plot()
    plt.show()
    

if __name__ == "__main__":
    test(model, test_dataloader)