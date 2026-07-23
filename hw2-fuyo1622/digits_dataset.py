# CSV-backed digit dataset shared by the DDPM and DANN tasks.
from torch.utils.data import Dataset
from torchvision import transforms
import csv
import os
from PIL import Image


# Load filenames and numeric labels from the split-specific CSV file.
class ImageDataset(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.file_label_list = []
        self.file_name_list = []
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ]
        )

        if mode == "train" or "val":
            with open(root + "/" + mode + ".csv", newline="") as csvfile:
                rows = csv.reader(csvfile)
                next(rows)
                for row in rows:
                    self.file_label_list.append(int(row[1]))
                    self.file_name_list.append(row[0])
        else:
            pass

    def __getitem__(self, idx):
        file_path = os.path.join(
            self.root + "/data",
            self.file_name_list[idx],
        )
        img = Image.open(file_path).convert("RGB")
        img = self.transform(img)
        label = self.file_label_list[idx]

        return img, label

    def __len__(self):
        return len(self.file_name_list)
