# Route each segment according to crop availability and face detection.
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


# Tag the training segments.
seg_name = []
seg_label = []
new_seg_file = [["seg_name", "ttm", "tag"]]
with open("./train.csv", newline="") as f:
    rows = csv.reader(f)
    _ = next(rows)
    for row in rows:
        seg_name.append(row[0])
        seg_label.append(int(row[1]))

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


# Iteration records a training tag as a side effect of loading each segment.
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
                image = face_recognition.load_image_file(
                    os.path.join(
                        "./videos",
                        videoname,
                        people + "_" + str(i) + ".png",
                    )
                )
                face_locations = face_recognition.face_locations(
                    image,
                    model="HOG",
                )
                if len(face_locations) == 1:
                    face += 1

        if len(data) == 0:
            new_seg_file.append([segname, str(label), "1"])
            return [], label, 0
        elif len(data) == 16:
            if face > 0:
                new_seg_file.append([segname, str(label), "3"])
            else:
                new_seg_file.append([segname, str(label), "2"])
            return torch.stack(data), label, face / len(data)
        else:
            if face > 0:
                new_seg_file.append([segname, str(label), "3"])
            else:
                new_seg_file.append([segname, str(label), "2"])
            extend_data = [
                data[math.floor(i * len(data) / 16)]
                for i in range(16)
            ]
            return torch.stack(extend_data), label, face / len(data)


videodataset = VideoDataset(seg_name, seg_label)
videodataloader = DataLoader(
    videodataset,
    batch_size=1,
    shuffle=False,
)
for batch in tqdm(videodataloader):
    pass
with open("train_tag.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(new_seg_file)


# Tag the test segments.
seg_name = []
seg_label = []
new_seg_file = [["seg_name", "ttm", "tag"]]
with open("./test.csv", newline="") as f:
    rows = csv.reader(f)
    _ = next(rows)
    for row in rows:
        seg_name.append(row[0])
        seg_label.append(int(row[1]))

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


# Repeat the same tagging procedure for the test segment table.
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
                image = face_recognition.load_image_file(
                    os.path.join(
                        "./videos",
                        videoname,
                        people + "_" + str(i) + ".png",
                    )
                )
                face_locations = face_recognition.face_locations(
                    image,
                    model="HOG",
                )
                if len(face_locations) == 1:
                    face += 1

        if len(data) == 0:
            new_seg_file.append([segname, str(label), "1"])
            return [], label, 0
        elif len(data) == 16:
            if face > 0:
                new_seg_file.append([segname, str(label), "3"])
            else:
                new_seg_file.append([segname, str(label), "2"])
            return torch.stack(data), label, face / len(data)
        else:
            if face > 0:
                new_seg_file.append([segname, str(label), "3"])
            else:
                new_seg_file.append([segname, str(label), "2"])
            extend_data = [
                data[math.floor(i * len(data) / 16)]
                for i in range(16)
            ]
            return torch.stack(extend_data), label, face / len(data)


videodataset = VideoDataset(seg_name, seg_label)
videodataloader = DataLoader(
    videodataset,
    batch_size=1,
    shuffle=False,
)
for batch in tqdm(videodataloader):
    pass
with open("test_tag.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(new_seg_file)
