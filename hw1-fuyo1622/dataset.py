# Dataset loaders and image transforms for HW1 classification and segmentation.
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import torch

# Classification training uses augmentation; evaluation stays deterministic.
p1_train_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

p1_test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# Classification labels are encoded in the filename prefix during training.
class ClassficationDataset(Dataset):
    def __init__(self, root, mode):
        file_name_list = os.listdir(root)
        self.mode = mode
        self.data_name_list = []
        self.label = []

        if mode == "train":
            self.transform = p1_train_transform
            for file_name in file_name_list:
                self.data_name_list.append(os.path.join(root, file_name))
                self.label.append(int(file_name.split("_")[0]))
        else:
            self.transform = p1_test_transform
            for file_name in file_name_list:
                self.data_name_list.append(os.path.join(root, file_name))

    def __getitem__(self, idx):
        if self.mode == "train" or self.mode == "val":
            file_path = self.data_name_list[idx]
            img = Image.open(file_path).convert("RGB")
            img = self.transform(img)
            label = self.label[idx]
            return img, label, file_path
        else:
            file_path = self.data_name_list[idx]
            img = Image.open(file_path).convert("RGB")
            img = self.transform(img)
            return img, file_path

    def __len__(self):
        return len(self.data_name_list)


# Segmentation samples pair each satellite image with its color-coded mask.
class SegmentationDataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.data = self.parse_data(root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            "img": sample[0],
            "label": sample[1],
            "id": sample[2],
        }

        return instance

    def collate_fn(self, samples):
        batch = {
            "img": torch.cat(
                [
                    self.transforms(sample["img"]).unsqueeze(0)
                    for sample in samples
                ]
            ),
            "label": torch.tensor([sample["label"] for sample in samples]),
            "id": [sample["id"] for sample in samples],
        }
        return batch

    def parse_data(self, data_path):
        file_list = os.listdir(data_path)
        file_list = set([file_name.split("_")[0] for file_name in file_list])
        data_list = []

        for i, file_name in enumerate(file_list):
            img_path = os.path.join(data_path, f"{file_name}_sat.jpg")
            label_path = os.path.join(data_path, f"{file_name}_mask.png")
            img = Image.open(img_path).convert("RGB")

            if self.mode == "train" or self.mode == "val":
                mask_img = np.array(Image.open(label_path).convert("RGB"))
                label = self.read_mask(mask_img)
            else:
                label = np.zeros(img.size[0:2])

            data_list.append([img, label, file_name])

        return data_list

    def read_mask(self, mask):
        # Decode the three thresholded RGB bits into the seven class IDs.
        mask = np.array(mask)
        label = np.empty(mask.shape[0:2], dtype=np.int32)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        label[mask == 3] = 0
        label[mask == 6] = 1
        label[mask == 5] = 2
        label[mask == 2] = 3
        label[mask == 1] = 4
        label[mask == 7] = 5
        label[mask == 0] = 6
        label[mask == 4] = 6

        return label
