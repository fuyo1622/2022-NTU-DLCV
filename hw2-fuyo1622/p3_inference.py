# Apply the DANN digit classifier to an unlabeled target-domain folder.
from digits_dataset import ImageDataset
from DANN_model import DANN

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib import tzip
from torch.autograd import Function
from PIL import Image
import csv
from torch.utils.data import Dataset, DataLoader
import argparse
from torchvision import models, transforms


# Test loader preserves filenames so predictions can be written to CSV.
class TestDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.file_name_list = os.listdir(root)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ]
        )

    def __getitem__(self, idx):
        file_path = os.path.join(
            self.root,
            self.file_name_list[idx],
        )
        img = Image.open(file_path).convert("RGB")
        img = self.transform(img)

        return img, self.file_name_list[idx]

    def __len__(self):
        return len(self.file_name_list)


def main(args):
    if not os.path.exists(os.path.dirname(args.pred_path)):
        os.makedirs(os.path.dirname(args.pred_path))

    device = args.device
    test_data_path = args.test_data_path
    test_dataset = TestDataset(test_data_path)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = DANN().to(device)

    # Select the checkpoint trained for the target domain named in the path.
    if "svhn" in test_data_path:
        ckpt = torch.load("p3_svhn_best_model.pth")
        model.load_state_dict(ckpt)
    else:
        ckpt = torch.load("p3_usps_best_model.pth")
        model.load_state_dict(ckpt)

    # Domain predictions are unused during target-class inference.
    predictions = {"filenames": [], "preds": []}
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            imgs, file_names = batch
            logits, _ = model(imgs.to(device), alpha=0)
            preds = logits.argmax(1)
            predictions["filenames"] += file_names
            predictions["preds"] += preds.tolist()

    pred_str = "image_name,label\n"
    for filename, pred in zip(
        predictions["filenames"],
        predictions["preds"],
    ):
        filename = filename.split("/")[-1]
        pred_str += "{},{}\n".format(filename, pred)

    with open(os.path.join(args.pred_path), "w") as f:
        f.write(pred_str)

    print("PREDICT DONE")







def _parse_args():
    parser = argparse.ArgumentParser(description="Script to train.")
    parser.add_argument(
        "-test_data_path",
        default="./digits/svhn/val",
        type=str,
    )
    parser.add_argument(
        "-pred_path",
        default="./digits/svhn/test_pred.csv",
        type=str,
    )
    parser.add_argument("-device", default="cuda", type=str)
    parser.add_argument("-batch_size", default=128, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
