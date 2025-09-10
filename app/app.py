import os
import io
import time
import torch
import random
import matplotlib
import customtkinter as ctk
from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt
from customtkinter import filedialog
from PIL import Image, ImageFont, ImageDraw

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import torchaudio

ctk.set_appearance_mode("dark")
matplotlib.use("Agg")

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
        self.img_transform = torchaudio.transforms.AmplitudeToDB(stype='magnitude', top_db=100)

        self.signals = []
        self.images = []

        for audio_path in self.filenames:
            signal, sr = torchaudio.load(audio_path) #get audio signal & sample rate
            signal = self._resample_if_necessary(signal, sr)
            if signal.shape[0] > 1: signal = torch.mean(signal, dim=0, keepdim=True) #change channels
            if signal.shape[1] > self.NUM_SAMPLES: signal = signal[:, :self.NUM_SAMPLES] #cut signal
            signal = self._right_pad_if_necessary(signal) #pad signal
            signal = self.transform(signal)

            # create image
            image = self.img_transform(signal)
            plt.imshow(image.squeeze(0).squeeze(0), aspect="auto", origin="lower", cmap="magma")
            plt.colorbar(label="dB")
            plt.ylabel("Freq", fontsize=14)
            plt.xlabel("Frame", fontsize=14)

            # change to PIL img
            with io.BytesIO() as f:
                plt.savefig(f, format="png", dpi=300, bbox_inches='tight')
                f.seek(0)
                image = Image.open(f).copy()
            plt.clf()
            plt.close('all')
            image = image.convert("RGB")
            image = ctk.CTkImage(image, size=(363, 235.2))

            self.signals.append(signal)
            self.images.append(image)

    
    # get length of data
    def __len__(self):
        return len(self.filenames)
    
    # get item
    def __getitem__(self, index):
        signal = self.signals[index]

        return signal
    
    def get_img(self, index):
        image = self.images[index]
        return image
    
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

#model
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


#app window
class App(ctk.CTk):
    # find device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #variables
    files = []
    results = []
    width = 864
    height = 614.4

    #create model
    input_len = 128
    hidden_size = 128
    num_layers = 2
    num_classes = 2

    model = CNNLSTM(input_len=input_len, hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers)

    def __init__(self):
        super().__init__()

        # window settings
        self.title("VoiceCheck: Cutting-edge AI voice detection")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(0, 0)
        self.configure(fg_color='#f5f5f7')
        current_path = os.path.dirname(os.path.realpath(__file__))

        #load weights
        self.model.load_state_dict(torch.load((current_path+"/CNNLSTM_VOICE_CV.pt"),map_location=torch.device(self.device)))

        #images
        self.landing_logo = ctk.CTkImage(Image.open(current_path+"/assets/VoiceCheck.png"), size=(405, 70.8))
        self.header_logo = ctk.CTkImage(Image.open(current_path+"/assets/VoiceCheck.png"), size=(363, 67.2))
        self.shadow_bg = ctk.CTkImage(Image.open(current_path+"/assets/shadow.png"), size=(514, 542.4))
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

        self.load_btn = ctk.CTkButton(master=self, image=self.custom_text("Load Audio Files", self.SEMIBOLD, "#ffffff", 28, "#007aff"), text=None, fg_color="#007aff", hover_color="#005FCC", width=204, height=97.8, corner_radius=27.6 ,command=self.next_page)
        self.load_btn.bind("<Enter>", lambda event, button=self.load_btn: button.configure(image=self.custom_text("Load Audio Files", self.SEMIBOLD, "#ffffff", 28, "#005FCC"), fg_color="#005FCC"))
        self.load_btn.bind("<Leave>", lambda event, button=self.load_btn: button.configure(image=self.custom_text("Load Audio Files", self.SEMIBOLD, "#ffffff", 28, "#007aff"), fg_color="#007aff"))
        self.load_btn.place(x=self.width//2, y=334.2, anchor="center")

        self.notice_text = ctk.CTkLabel(master=self, image=self.custom_text("*Supported: .wav, .ogg, .mp3", self.REGULAR, "#000000", 29, "#f5f5f7"), fg_color="transparent", text=None)
        self.notice_text.place(x=61.2, y=511.8, anchor="nw")

        self.info_btn = ctk.CTkButton(master=self, image=self.info_img, fg_color="transparent", hover_color="#f5f5f7", text=None, width=41.4, height=41.4, command=self.create_window)
        self.info_btn.place(x=802.8, y=511.8, anchor="ne")

        #analysis page
        self.header = ctk.CTkFrame(self, height=72, width=self.width, fg_color="#ffffff", corner_radius=0)
        self.sidebar = ctk.CTkFrame(self, height=542.4, width=350, fg_color="#f9fafb", corner_radius=0)
        self.content = ctk.CTkFrame(self, height=542.4, width=514, fg_color="#f9fafb", corner_radius=0)

        self.file_count = ctk.CTkLabel(self.header, image=self.custom_text("Files Loaded: 0", self.SEMIBOLD, "#4a4a4a", 42, "#ffffff"), height=72, text=None, fg_color="transparent")
        self.small_logo = ctk.CTkLabel(self.header, image=self.header_logo, width=363, height=72, text=None, fg_color="transparent")

        self.sound_list = CTKListBox(self.sidebar, width=350, height=301.2, fg_color="#f9fafb", corner_radius=0)
        self.upload_btn = ctk.CTkButton(self.sidebar, image=self.custom_text("Upload", self.SEMIBOLD, "#ffffff", 32, "#007aff"), text=None, fg_color="#007aff", hover_color="#005FCC", width=174, height=75, corner_radius=27.6 ,command=self.upload_files)
        self.upload_btn.bind("<Enter>", lambda event, button=self.upload_btn: button.configure(image=self.custom_text("Upload", self.SEMIBOLD, "#ffffff", 32, "#005FCC"), fg_color="#005FCC"))
        self.upload_btn.bind("<Leave>", lambda event, button=self.upload_btn: button.configure(image=self.custom_text("Upload", self.SEMIBOLD, "#ffffff", 32, "#007aff"), fg_color="#007aff"))

        self.bg_label = ctk.CTkLabel(self.content, image=self.shadow_bg, width=542.4, height=514, fg_color="transparent", text=None, corner_radius=0)
        self.specto_img = ctk.CTkLabel(self.content, fg_color="white", width=363, height=235.2, text=None)
        self.play_btn = ctk.CTkButton(self.content, image=self.custom_text("Play", self.SEMIBOLD, "#ffffff", 25, "#34c759"), text=None, fg_color="#34c759", hover_color="#28A745", width=117, height=48, corner_radius=27.6, command=lambda: self.playsound(self.files[self.sound_list.chosen]))
        self.play_btn.bind("<Enter>", lambda event, button=self.play_btn: button.configure(image=self.custom_text("Play", self.SEMIBOLD, "#ffffff", 25, "#28A745"), text=None, fg_color="#28A745"))
        self.play_btn.bind("<Leave>", lambda event, button=self.play_btn: button.configure(image=self.custom_text("Play", self.SEMIBOLD, "#ffffff", 25, "#34c759"), text=None, fg_color="#34c759"))
        self.result_text_label = ctk.CTkLabel(self.content, image=self.custom_text("Result:", self.SEMIBOLD, "#000000", 32, "#ffffff", pad_right=79), width=151, height=30, text=None, fg_color="white")
        self.result_value_label = ctk.CTkLabel(self.content, width=96, height=30, text=None, fg_color="white")
        self.result_spacer_label = ctk.CTkLabel(self.content, width=151, height=30, text=None, fg_color="white")
        self.confidence_text_label = ctk.CTkLabel(self.content, image=self.custom_text("Confidence:", self.SEMIBOLD, "#000000", 32, "#ffffff"), width=151, height=30, text=None, fg_color="white")
        self.confidence_value_label = ctk.CTkLabel(self.content, width=96, height=30, text=None, fg_color="white")
        self.confidence_spacer_label = ctk.CTkLabel(self.content, width=151, height=30, text=None, fg_color="white")
        self.prev_btn = ctk.CTkButton(self.content, image=self.custom_text("< Previous", self.SEMIBOLD, "#ffffff", 20, "#d1d1d6"), text=None, fg_color="#d1d1d6", hover_color="#E5E5EA", width=141, height=63.6, corner_radius=27.6, command=lambda: self.sound_list.choose_file((self.sound_list.chosen-1)%len(self.files)))
        self.prev_btn.bind("<Enter>", lambda event, button=self.prev_btn: button.configure(image=self.custom_text("< Previous", self.SEMIBOLD, "#ffffff", 20, "#E5E5EA"), text=None, fg_color="#E5E5EA"))
        self.prev_btn.bind("<Leave>", lambda event, button=self.prev_btn: button.configure(image=self.custom_text("< Previous", self.SEMIBOLD, "#ffffff", 20, "#d1d1d6"), text=None, fg_color="#d1d1d6"))
        self.next_btn = ctk.CTkButton(self.content, image=self.custom_text("Next >", self.SEMIBOLD, "#ffffff", 25, "#d1d1d6"), text=None, fg_color="#d1d1d6", hover_color="#E5E5EA", width=141, height=63.6, corner_radius=27.6, command=lambda: self.sound_list.choose_file((self.sound_list.chosen+1)%len(self.files)))
        self.next_btn.bind("<Enter>", lambda event, button=self.next_btn: button.configure(image=self.custom_text("Next >", self.SEMIBOLD, "#ffffff", 25, "#E5E5EA"), text=None, fg_color="#E5E5EA"))
        self.next_btn.bind("<Leave>", lambda event, button=self.next_btn: button.configure(image=self.custom_text("Next >", self.SEMIBOLD, "#ffffff", 25, "#d1d1d6"), text=None, fg_color="#d1d1d6"))
    
    def playsound(self, file):
        sound = AudioSegment.from_file(file)[:2000]
        play(sound)
    
    def custom_text(self, text, font, color, fontsize, bgcolor, anchor="lt", pad_right=0):
        #load font
        font = ImageFont.truetype(font=font, size=fontsize)

        #get size
        dummy_image = Image.new(mode="RGBA", size=(1, 1))
        dummy_draw = ImageDraw.Draw(dummy_image)
        text = text.split("\n") #seperate by newline (enter)
        left, top, right, bottom = dummy_draw.textbbox((0, 0), text=max(text, key=len), font=font, anchor=anchor)
        width = right - left + 10 + pad_right#10px padding
        height = (bottom - top + 10) * len(text)

        #create img
        image = Image.new(mode="RGBA", size=(width, height), color=bgcolor)
        draw = ImageDraw.Draw(image)
        draw.fontmode = "L"
        for i, line in enumerate(text):
            draw.text(xy=(5, 5+height/len(text)*i), text=line, font=font, fill=color, anchor=anchor)
        image = ctk.CTkImage(image, size=(width,height))
        return image
    
    def create_window(self):
        info = info_window(self, self.SEMIBOLD, self.REGULAR)
    
    def next_page(self):
        self.upload_files()
        self.big_logo.place_forget()
        self.tag_line.place_forget()
        self.load_btn.place_forget()
        self.notice_text.place_forget()
        self.info_btn.place_forget()

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

        self.header.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.sidebar.grid(row=1, column=0, sticky="nsew")
        self.content.grid(row=1, column=1, sticky="nsew")

        self.header.grid_columnconfigure(0, weight=0)
        self.header.grid_columnconfigure(1, weight=1)
        self.header.grid_columnconfigure(2, weight=0)
        self.file_count.grid(row=0, column=0, sticky="nsw")
        self.small_logo.grid(row=0, column=2, sticky="nse")

        self.sidebar.grid_columnconfigure(0, weight=0)
        self.sidebar.rowconfigure(0, weight=1)
        self.sidebar.rowconfigure(1, weight=0)
        self.sidebar.rowconfigure(2, weight=1)
        self.sidebar.rowconfigure(3, weight=0)
        self.sidebar.rowconfigure(4, weight=1)
        self.sound_list.grid(row=1, column=0)
        self.upload_btn.grid(row=3, column=0)

        self.content.rowconfigure(0, weight=0)
        self.content.rowconfigure(1, weight=0)
        self.content.rowconfigure(2, weight=0)
        self.content.rowconfigure(3, weight=0)
        self.content.rowconfigure(4, weight=0)
        self.content.rowconfigure(5, weight=0)
        self.content.rowconfigure(6, weight=0)
        self.content.columnconfigure(0, weight=1)
        self.content.columnconfigure(1, weight=0)
        self.content.columnconfigure(2, weight=0)
        self.content.columnconfigure(3, weight=0)
        self.content.columnconfigure(4, weight=1)
        self.bg_label.grid(row=0, column=0, sticky="nsew", rowspan=7, columnspan=5)
        self.specto_img.grid(row=2, column=1, columnspan=3)
        self.play_btn.grid(row=3, column=2, pady=20)
        self.result_text_label.grid(row=4, column=1)
        self.result_value_label.grid(row=4, column=2)
        self.result_spacer_label.grid(row=4, column=3)
        self.confidence_text_label.grid(row=5, column=1)
        self.confidence_value_label.grid(row=5, column=2)
        self.confidence_spacer_label.grid(row=5, column=3)
        self.prev_btn.grid(row=6, column=1, pady=(20, 30))
        self.next_btn.grid(row=6, column=3, pady=(20, 30))
    
    def upload_files(self):
        self.files = filedialog.askopenfilenames(filetypes=[("Audio Files", "*.wav *.ogg *.mp3")])
        self.file_count.configure(image=self.custom_text(f"Files Loaded: {len(self.files)}", self.SEMIBOLD, "#4a4a4a", 42, "#ffffff"))

        #get spectograms
        self.dataset = CreateDataset(self.files, self.device)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        #get results
        self.results = []
        softmax = nn.Softmax(dim=1)
        self.model.eval()
        with torch.no_grad():
            for batch, image in enumerate(self.dataloader):
                image = image.to(self.device)
                x = self.model(image)
                x = softmax(x).squeeze().tolist()
                self.results.append(x)

        self.sound_list.load_items(values=self.files)
    
    def get_results(self, index):
        img = self.dataset.get_img(index)
        self.specto_img.configure(image=img)

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

        #widgets
        self.what_header = ctk.CTkLabel(master=self, image=master.custom_text("What is VoiceCheck", bold_font, "#000000", 25.2, "#f5f5f7"), text=None, fg_color="transparent")
        self.what_header.place(x=18.2, y=10, anchor="nw")

        self.what_text = ctk.CTkLabel(master=self, image=master.custom_text(
            "VoiceCheck analyzes audio files to determine whether the \nvoice is human or AI-generated. It uses a deep learning \nmodel trained on real and synthetic speech patterns to make \naccurate classifications. Your files never leave your computer. \nAI analysis happens locally using on-device AI."
            , regular_font, "#000000", 16.8, "#f5f5f7"), text=None, fg_color="transparent")
        self.what_text.place(x=18.2, y=45.6, anchor="nw")

        self.how_header = ctk.CTkLabel(master=self, image=master.custom_text("How to use VoiceCheck", bold_font, "#000000", 25.2, "#f5f5f7"), text=None, fg_color="transparent")
        self.how_header.place(x=18.2, y=180.6, anchor="nw")

        self.how_text = ctk.CTkLabel(master=self, image=master.custom_text(
            "Simply click the “Load Audio Files” button and choose files \nyou want. Navigate between files using “previous” and “next” \nbuttons, or simply choose files using the sidebar."
            , regular_font, "#000000", 16.8, "#f5f5f7"), text=None, fg_color="transparent")
        self.how_text.place(x=18.2, y=216.2, anchor="nw")

#listbox

class CTKListBox(ctk.CTkScrollableFrame):
    def __init__(self, master, width=350, height=301.2, fg_color="#f9fafb", corner_radius=0):
        super().__init__(master=master, width=width, height=height, fg_color=fg_color, corner_radius=corner_radius)
        self.app = master.master

        self.file_names = []
        self.index_labels = []
        self.indicators = []
        self.chosen = None
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=0)
        self.columnconfigure(2, weight=2)
    
    def load_items(self, values):
        for widget in self.file_names: widget.grid_forget()
        for widget in self.indicators: widget.grid_forget()
        self.file_names, self.indicators = [], []
        values = [value.split("/")[-1] for value in values]
        for i, value in enumerate(values):
            indicator = ctk.CTkLabel(self, width=91, height=27, text=None, fg_color="transparent")
            widget = ctk.CTkButton(self, height=27, image=self.app.custom_text(f"{i+1:3d}. {value}", self.app.REGULAR, "#000000", 28, "#f9fafb"), text=None, fg_color="transparent", hover=False, command=lambda index=i: self.choose_file(index))
            self.indicators.append(indicator)
            self.file_names.append(widget)
            indicator.grid(row=i, column=0)
            widget.grid(row=i, column=1)
        if len(values) > 0:
            self.indicators[0].configure(image = self.app.custom_text(">", self.app.SEMIBOLD, "#000000", 28, "#f9fafb"))
            self.chosen = 0
            self.app.get_results(self.chosen)
            self.update_values(self.chosen)
        else: self.chosen = None
    
    def choose_file(self, index):
        if self.chosen != index:
            self.indicators[self.chosen].configure(image = None)
            self.chosen = index
            self.app.get_results(self.chosen)
            self.indicators[self.chosen].configure(image = self.app.custom_text(">", self.app.SEMIBOLD, "#000000", 28, "#f9fafb"))
            self.update_values(self.chosen)

    def update_values(self, index):
        percent = max(self.app.results[index])
        result = self.app.results[index].index(percent)
        if result == 0:
            result = "AI"
        else:
            result = "Human"
            
        self.app.result_value_label.configure(image=self.app.custom_text(result, self.app.REGULAR, "#FF3B30", 32, "white"))
        self.app.confidence_value_label.configure(image=self.app.custom_text(str(int(percent*100))+"%", self.app.REGULAR, "#FF3B30", 32, "white"))

# Run application
if __name__ == "__main__":
    # create app
    app = App()
    
    app.mainloop()
