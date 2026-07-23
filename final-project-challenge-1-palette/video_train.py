# Train the video classifier on fixed-length face-crop sequences.
import torch
import cv2
from PIL import Image
import csv
import os
from torchvision.models.video import mvit_v2_s, s3d
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision.io import write_video
import torch.nn as nn
import torch.utils.data as data
import math
import face_recognition
from matplotlib import pyplot as plt
import statistics
from sklearn.model_selection import KFold
import pytorchvideo
import torchvision


seg_name = []
seg_label = []

with open("./train_tag.csv", newline="") as f:
    rows = csv.reader(f)
    _ = next(rows)
    for row in rows:
        if row[2] == "2" or row[2] == "3":
            seg_name.append(row[0])
            seg_label.append(int(row[1]))

# Apply the normalization expected by the pretrained video backbone.
train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.43216, 0.394666, 0.37645],
            std=[0.22803, 0.22145, 0.216989],
        ),
    ]
)


# Sample or repeat available crops to produce exactly 16 frames.
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
            return [], label, 0
        elif len(data) == 16:
            return torch.stack(data), label, face / len(data)
        else:
            extend_data = [
                data[math.floor(i * len(data) / 16)]
                for i in range(16)
            ]
            return torch.stack(extend_data), label, face / len(data)


model.to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
videodataset = VideoDataset(seg_name, seg_label)
videodataloader = DataLoader(
    videodataset,
    batch_size=1,
    shuffle=False,
)
device = "cuda"
# Evaluate training progress with five-fold splits.
skf = KFold(n_splits=5)

best_acc = 0.0
for epoch in range(10):
    total_train_acc = 0
    total_train_loss = 0
    total_val_acc = 0
    total_val_loss = 0
    for fold_i, (train_ids, val_ids) in enumerate(skf.split(videodataset)):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_loader = DataLoader(
            videodataset,
            batch_size=32,
            sampler=train_sampler,
        )
        val_loader = DataLoader(
            videodataset,
            batch_size=32,
            sampler=val_sampler,
        )
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            inputs, labels, _ = data
            inputs, labels = (
                inputs.permute(0, 2, 1, 3, 4).to(device),
                labels.to(device),
            )
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

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                inputs, labels, _ = data
                inputs, labels = (
                    inputs.permute(0, 2, 1, 3, 4).to(device),
                    labels.to(device),
                )
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
        print(train_acc, val_acc)

    print(
        (
            "[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | "
            "Val Acc: {:3.6f} loss: {:3.6f}"
        ).format(
            epoch + 1,
            10,
            total_train_acc / len(videodataset) / 4,
            total_train_loss / len(train_loader) / 5,
            total_val_acc / len(videodataset),
            total_val_loss / len(val_loader) / 5,
        )
    )
    if total_val_acc > best_acc:
        best_acc = total_val_acc
        torch.save(
            model.state_dict(),
            "./img_model/" + str(epoch) + ".pth",
        )
        print(
            "saving model with acc {:.3f}".format(
                best_acc / len(videodataset)
            )
        )
