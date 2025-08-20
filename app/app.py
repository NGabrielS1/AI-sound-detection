import os
import io
import time
import torch
import random
import customtkinter as ctk
import matplotlib.pyplot as plt
from customtkinter import filedialog

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import torchaudio

ctk.set_appearance_mode("dark")

#dataset class
class CreateDataset(Dataset):

    # initialize dataset variables
    def __init__(self, sounds: list, device):
        # variables
        self.TARGET_SAMPLE_RATE = 22050
        self.NUM_SAMPLES = 44100

        self.device = device
        self.filenames = sounds
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.TARGET_SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
    
    # get length of data
    def __len__(self):
        return len(self.filenames)
    
    # get item
    def __getitem__(self, index):
        # get audio path
        audio_path = self.filenames[index]

        signal, sr = torchaudio.load(audio_path) #get audio signal & sample rate
        signal = self._resample_if_necessary(signal, sr)
        if signal.shape[0] > 1: signal = torch.mean(signal, dim=0, keepdim=True) #change channels
        if signal.shape[1] > self.NUM_SAMPLES: signal = signal[:, :self.NUM_SAMPLES] #cut signal
        signal = self._right_pad_if_necessary(signal) #pad signal
        signal = self.transform(signal)

        return signal
    
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

class App(ctk.CTk):
    # find device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #variables
    file = 0
    files = []
    width = 1440
    height = 1024

    def __init__(self):
        super().__init__()

        # window settings
        self.title("VoiceCheck: Cutting-edge AI voice detection")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(0, 0)

# Run application
if __name__ == "__main__":
    # create app
    app = App()
    
    app.mainloop()
