import os
import io
import time
import torch
import random
import customtkinter as ctk
import matplotlib.pyplot as plt
from customtkinter import filedialog
from PIL import Image, ImageFont, ImageDraw

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

#app window
class App(ctk.CTk):
    # find device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #variables
    file = 0
    files = []
    width = 864
    height = 614.4

    def __init__(self):
        super().__init__()

        # window settings
        self.title("VoiceCheck: Cutting-edge AI voice detection")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(0, 0)
        self.configure(fg_color='#f5f5f7')
        current_path = os.path.dirname(os.path.realpath(__file__))

        #images
        self.landing_logo = ctk.CTkImage(Image.open(current_path+"/assets/VoiceCheck.png"), size=(405, 70.8))
        self.info_img = ctk.CTkImage(Image.open(current_path+"/assets/info.png"), size=(41.4, 41.4))

        #fonts
        self.REGULAR = current_path+"/assets/Inter_28pt-Regular.ttf"
        self.ITALIC = current_path+"/assets/Inter_28pt-Italic.ttf"
        self.BOLD = current_path+"/assets/Inter_28pt-Bold.ttf"
        self.SEMIBOLD = current_path+"/assets/Inter_28pt-SemiBold.ttf"

        #landing page
        self.big_logo = ctk.CTkLabel(master=self, image=self.landing_logo, fg_color="transparent", text=None)
        self.big_logo.place(x=self.width//2, y=91.8, anchor="center")

        self.tag_line = ctk.CTkLabel(master=self, image=self.custom_text("Determine if your audio is human or AI in seconds.", self.ITALIC, "#000000", 28, "#f5f5f7"), text=None, fg_color="transparent")
        self.tag_line.place(x=self.width//2, y=180, anchor="center")

        self.load_btn = ctk.CTkButton(master=self, image=self.custom_text("Load Audio Files", self.SEMIBOLD, "#ffffff", 28, "#007aff"), text=None, fg_color="#007aff", hover_color="#005FCC", width=204, height=97.8, corner_radius=27.6)
        self.load_btn.bind("<Enter>", lambda event, button=self.load_btn: button.configure(image=self.custom_text("Load Audio Files", self.SEMIBOLD, "#ffffff", 28, "#005FCC"), fg_color="#005FCC"))
        self.load_btn.bind("<Leave>", lambda event, button=self.load_btn: button.configure(image=self.custom_text("Load Audio Files", self.SEMIBOLD, "#ffffff", 28, "#007aff"), fg_color="#007aff"))
        self.load_btn.place(x=self.width//2, y=334.2, anchor="center")

        self.notice_text = ctk.CTkLabel(master=self, image=self.custom_text("*Supported: .wav, .ogg, .mp3", self.REGULAR, "#000000", 29, "#f5f5f7"), fg_color="transparent", text=None)
        self.notice_text.place(x=61.2, y=511.8, anchor="nw")

        self.info_btn = ctk.CTkButton(master=self, image=self.info_img, fg_color="transparent", hover_color="#f5f5f7", text=None, width=41.4, height=41.4, command=self.create_window)
        self.info_btn.place(x=802.8, y=511.8, anchor="ne")

    
    def custom_text(self, text, font, color, fontsize, bgcolor, anchor="lt"):
        #load font
        font = ImageFont.truetype(font=font, size=fontsize)

        #get size
        dummy_image = Image.new(mode="RGBA", size=(1, 1))
        dummy_draw = ImageDraw.Draw(dummy_image)
        left, top, right, bottom = dummy_draw.textbbox((0, 0), text, font=font, anchor=anchor)
        width = right - left + 10 #10px padding
        height = bottom - top + 10

        #create img
        image = Image.new(mode="RGBA", size=(width, height), color=bgcolor)
        draw = ImageDraw.Draw(image)
        draw.fontmode = "L"
        text = text.split("\n") #seperate by newline (enter)
        for i, line in enumerate(text):
            draw.text(xy=(5, 5+height*i), text=line, font=font, fill=color, anchor=anchor)
        image = ctk.CTkImage(image, size=(width,height))
        return image
    
    def create_window(self):
        info = info_window(self, self.SEMIBOLD, self.REGULAR)

#info window
class info_window(ctk.CTkToplevel):
    width = 500
    height = 300

    def __init__(self, master, bold_font, regular_font):
        super().__init__(master=master)
        self.title("Information")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(0, 0)
        self.configure(fg_color='#f5f5f7')

        #text


# Run application
if __name__ == "__main__":
    # create app
    app = App()
    
    app.mainloop()
