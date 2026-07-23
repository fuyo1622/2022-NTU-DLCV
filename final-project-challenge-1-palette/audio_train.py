# Train the MFCC-based classifier used for audio-tagged segments.
import torch
import cv2
from PIL import Image
import csv
import os
from torchvision.models.video import mvit_v2_s
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio.transforms as transforms
from torchvision.io import write_video
import torch.nn as nn
import math
import face_recognition
from matplotlib import pyplot as plt
import statistics
import random
import torchaudio
from sklearn.model_selection import KFold
from torchsummary import summary


torch.set_printoptions(profile="full")
seg_name = []
seg_label = []
# Convert fixed-length stereo waveforms into flattened MFCC features.
transform = transforms.MFCC(
    sample_rate=16000,
    n_mfcc=256,
    melkwargs={
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 256,
        "mel_scale": "htk",
    },
)

with open("./train_tag.csv", newline="") as f:
    rows = csv.reader(f)
    _ = next(rows)
    for row in rows:
        if row[2] != "1":
            seg_name.append(row[0])
            seg_label.append(int(row[1]))


# Load, pad or crop, and transform one segment-level WAV file.
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
            "./train_seg_audio/" + segname + ".wav"
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

        return torch.flatten(seg_audio), label

        return seg_audio, label


# Fully connected block used repeatedly by the audio classifier.
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


# Stack dense blocks before the final two-class prediction layer.
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


audiodataset = VideoDataset(seg_name, seg_label)
audiodataloader = DataLoader(
    audiodataset,
    batch_size=1,
    shuffle=False,
)

model = Classifier01().to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
# Reuse five folds inside each epoch for training and validation.
skf = KFold(n_splits=5)
device = "cuda"

best_acc = 0.0
for epoch in range(80):
    total_train_acc = 0
    total_train_loss = 0
    total_val_acc = 0
    total_val_loss = 0
    for fold_i, (train_ids, val_ids) in enumerate(skf.split(audiodataset)):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_loader = DataLoader(
            audiodataset,
            batch_size=32,
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            audiodataset,
            batch_size=32,
            sampler=val_sampler,
        )

        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        # Training.
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            _, train_pred = torch.max(outputs, 1)
            batch_loss.backward()
            optimizer.step()

            train_acc += (
                (train_pred.cpu() == labels.cpu()).sum().item()
            )
            train_loss += batch_loss.item()

        # Validation.
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, val_pred = torch.max(outputs, 1)

                val_acc += (
                    (val_pred.cpu() == labels.cpu()).sum().item()
                )
                val_loss += batch_loss.item()

        total_train_acc += train_acc
        total_train_loss += train_loss
        total_val_acc += val_acc
        total_val_loss += val_loss

    print(
        (
            "[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | "
            "Val Acc: {:3.6f} loss: {:3.6f}"
        ).format(
            epoch + 1,
            80,
            total_train_acc / len(audiodataset) / 4,
            total_train_loss / len(train_loader) / 5,
            total_val_acc / len(audiodataset),
            total_val_loss / len(val_loader) / 5,
        )
    )

    torch.save(
        model.state_dict(),
        "./audio_model/" + str(epoch) + ".pth",
    )
