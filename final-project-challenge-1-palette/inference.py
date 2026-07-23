# Combine audio and video classifiers into the final TTM prediction CSV.
import torch
import cv2
from PIL import Image
import csv
import os
from torchvision.models.video import mvit_v2_s
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio.transforms as audio_transforms
import torchvision.transforms as vision_transforms
from torchvision.io import write_video
import torch.nn as nn
import math
import face_recognition
from matplotlib import pyplot as plt
import statistics
import random
import torchaudio
import torchvision
from sklearn.model_selection import KFold


random.seed(1121)
torch.set_printoptions(profile="full")
seg_name_1 = []
seg_label_1 = []
seg_name_2 = []
seg_label_2 = []
seg_name_3 = []
seg_label_3 = []
transform = audio_transforms.MFCC(
    sample_rate=16000,
    n_mfcc=256,
    melkwargs={
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 256,
        "mel_scale": "htk",
    },
)

# Split segments according to their preprocessing tag.
with open("./test_tag.csv", newline="") as f:
    rows = csv.reader(f)
    _ = next(rows)
    for row in rows:
        if row[2] == "1":
            seg_name_1.append(row[0])
            seg_label_1.append(int(row[1]))
        elif row[2] == "2":
            seg_name_2.append(row[0])
            seg_label_2.append(int(row[1]))
        else:
            seg_name_3.append(row[0])
            seg_label_3.append(int(row[1]))


# Audio dataset for segments assigned to the waveform-based branch.
class VideoDataset(Dataset):
    def __init__(self, seg_name, seg_label):
        super().__init__()

        self.segnames = seg_name
        self.labels = seg_label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        segname = self.segnames[idx]
        seg_audio, seg_sample_rate = torchaudio.load(
            "./test_seg_audio/" + segname + ".wav"
        )

        if seg_audio.shape[1] > 127 * 512:
            seg_audio = seg_audio[:, : 127 * 512]
        else:
            pad_begin_len = random.randint(
                0,
                127 * 512 - seg_audio.shape[1],
            )
            pad_end_len = (
                127 * 512 - seg_audio.shape[1] - pad_begin_len
            )
            pad_begin = torch.zeros((2, pad_begin_len))
            pad_end = torch.zeros((2, pad_end_len))
            seg_audio = torch.cat(
                (pad_begin, seg_audio, pad_end),
                1,
            )

        seg_audio = transform(seg_audio)

        return segname, torch.flatten(seg_audio), label


class BasicBlock01(nn.Module):
    def __init__(self, input_dim, output_dim, p=0.5):
        super(BasicBlock01, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
            nn.Dropout(p),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier01(nn.Module):
    def __init__(
        self,
        input_dim=65536,
        output_dim=2,
        hidden_layers=3,
        hidden_dim=2048,
    ):
        super(Classifier01, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock01(input_dim, hidden_dim),
            *[
                BasicBlock01(hidden_dim, hidden_dim)
                for _ in range(hidden_layers)
            ],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


# Run audio-model inference for tag-1 segments.
audiodataset = VideoDataset(seg_name_1, seg_label_1)
audiodataloader = DataLoader(
    audiodataset,
    batch_size=1,
    shuffle=False,
)
model = Classifier01().to("cuda")

model.load_state_dict(torch.load("./audio_model.pth"))
device = "cuda"
result = [["Id", "Predicted"]]
model.eval()
with torch.no_grad():
    for i, data in enumerate(tqdm(audiodataloader)):
        segname, inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, val_pred = torch.max(outputs, 1)
        result.append([segname[0], val_pred.item()])


train_transform = vision_transforms.Compose(
    [
        vision_transforms.Resize(256),
        vision_transforms.CenterCrop(224),
        vision_transforms.ToTensor(),
        vision_transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989],
        ),
    ]
)


# Video dataset builds a fixed 16-frame face-crop sequence per segment.
class VideoDataset(Dataset):
    def __init__(self, seg_name, seg_label):
        super().__init__()

        self.segnames = seg_name
        self.labels = seg_label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        segname = self.segnames[idx]
        seg_info = segname.split("_")
        videoname = seg_info[0]
        people = seg_info[1]
        start_f = int(seg_info[2])
        end_f = int(seg_info[3])
        data = []
        if end_f - start_f < 15:
            use_f = list(range(start_f, end_f + 1))
        else:
            use_f = [
                int(start_f + i * (end_f - start_f) / 15)
                for i in range(16)
            ]
        face = 0
        for i in use_f:
            if os.path.isfile(
                os.path.join(
                    "./videos",
                    videoname,
                    people + "_" + str(i) + ".png",
                )
            ):
                img = Image.open(
                    os.path.join(
                        "./videos",
                        videoname,
                        people + "_" + str(i) + ".png",
                    )
                ).convert("RGB")
                img = train_transform(img)
                data.append(img)

        if len(data) == 0:
            return segname, [], label, 0
        elif len(data) == 16:
            return segname, torch.stack(data), label, face / len(data)
        else:
            extend_data = [
                data[math.floor(i * len(data) / 16)]
                for i in range(16)
            ]
            return (
                segname,
                torch.stack(extend_data),
                label,
                face / len(data),
            )


# Run video-model inference for tag-2 segments.
videodataset = VideoDataset(seg_name_2, seg_label_2)
videodataloader = DataLoader(
    videodataset,
    batch_size=1,
    shuffle=False,
)
model = torchvision.models.video.s3d(weights="DEFAULT")
model.classifier[1] = nn.Conv3d(
    1024,
    2,
    kernel_size=(1, 1, 1),
    stride=(1, 1, 1),
)
model.to("cuda")
model.load_state_dict(torch.load("./img_model.pth"))
model.eval()
with torch.no_grad():
    for i, data in enumerate(tqdm(videodataloader)):
        segname, inputs, labels, _ = data
        inputs, labels = (
            inputs.permute(0, 2, 1, 3, 4).to(device),
            labels.to(device),
        )
        outputs = model(inputs)
        _, val_pred = torch.max(outputs, 1)
        result.append([segname[0], val_pred.item()])


# Run video-model inference for tag-3 segments.
videodataset = VideoDataset(seg_name_3, seg_label_3)
videodataloader = DataLoader(
    videodataset,
    batch_size=1,
    shuffle=False,
)

model = torchvision.models.video.s3d(weights="DEFAULT")
model.classifier[1] = nn.Conv3d(
    1024,
    2,
    kernel_size=(1, 1, 1),
    stride=(1, 1, 1),
)
model.to("cuda")
model.load_state_dict(torch.load("./img_model.pth"))
model.eval()
with torch.no_grad():
    for i, data in enumerate(tqdm(videodataloader)):
        segname, inputs, labels, _ = data
        inputs, labels = (
            inputs.permute(0, 2, 1, 3, 4).to(device),
            labels.to(device),
        )
        outputs = model(inputs)
        _, val_pred = torch.max(outputs, 1)
        result.append([segname[0], val_pred.item()])

with open("pred.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(result)
