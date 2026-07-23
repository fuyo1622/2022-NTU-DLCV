# Train and validate the HW1 semantic-segmentation models.
import torch
import os
import torchvision
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import imageio
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from model import vgg16fcn32
import argparse
from dataset import SegmentationDataset


def miou_score(pred, labels, store):
    # Track predicted, ground-truth, and intersection pixels per class.
    pred = pred.detach().numpy()
    labels = labels.cpu().detach().numpy()
    for i in range(6):
        store[i][0] += np.sum(pred == i)
        store[i][1] += np.sum(labels == i)
        store[i][2] += np.sum((pred == i) * (labels == i))
    return store


def main(args):
    device = args.device
    # Both loaders use the dataset's custom image-and-mask collator.
    print("LOAD DATASET")

    train_dataset = SegmentationDataset(args.train_data_path, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dataset = SegmentationDataset(args.valid_data_path, "val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    print("LOAD DATASET DONE")

    # Choose between the FCN-32s baseline and the DeepLabV3 alternative.
    print("SET MODEL")

    num_classes = 7

    if args.model_for_question == 1:
        model = vgg16fcn32(num_classes).to(device)
    else:
        model = models.segmentation.deeplabv3_resnet101(
            num_classes=num_classes
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    n_epochs = args.epochs
    best_acc = -1
    best_state_dict = None
    save_path = args.save_model_path
    print("SET MODEL DONE")

    print("TRAINING")

    for epoch in range(n_epochs):
        # Optimize the model and accumulate the statistics needed for mIoU.
        model.train()
        train_loss = 0
        train_accs = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        for batch in tqdm(train_loader):
            img = batch["img"].to(device)
            label = batch["label"].long().to(device)
            logit = model(img)["out"]
            batch_loss = criterion(logit, label)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            pred = logit.argmax(1).cpu()
            train_accs = miou_score(pred, label, train_accs)

        loss = loss / len(train_loader)
        train_acc = 0
        for i in range(6):
            train_acc += train_accs[i][2] / (
                train_accs[i][0] + train_accs[i][1] - train_accs[i][2]
            )
        train_acc /= 6

        print(
            f"[ TRAIN | {epoch + 1:03d}/{n_epochs:03d} ] "
            f"LOSS = {loss:.5f}, ACC = {train_acc:.5f}"
        )

        # Recompute the same mIoU statistics on the validation split.
        model.eval()
        loss = 0
        valid_accs = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        for batch in tqdm(val_loader):
            img = batch["img"].to(device)
            label = batch["label"].long().to(device)
            with torch.no_grad():
                logit = model(img)["out"]

            loss += batch_loss.item()
            pred = logit.argmax(1).cpu()
            valid_accs = miou_score(pred, label, valid_accs)

        loss = loss / len(val_loader)
        valid_acc = 0
        for i in range(6):
            valid_acc += valid_accs[i][2] / (
                valid_accs[i][0] + valid_accs[i][1] - valid_accs[i][2]
            )
        valid_acc /= 6

        print(
            f"[ VALID | {epoch + 1:03d}/{n_epochs:03d} ] "
            f"LOSS = {loss:.5f}, ACC = {valid_acc:.5f}"
        )

        # Save the best validation checkpoint plus fixed training milestones.
        if valid_acc.item() >= best_acc:
            best_acc = valid_acc
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, save_path)
            print(
                "SAVE MODEL FOR EPOCH:",
                epoch + 1,
                "ACC=",
                best_acc,
            )
        if (
            epoch == 0
            or epoch + 1 == n_epochs
            or epoch + 1 == n_epochs // 2
        ):
            torch.save(model.state_dict(), str(epoch + 1) + "model.pth")
    print("TRAIN DONE")


def _parse_args():
    parser = argparse.ArgumentParser(description="Script to train.")
    parser.add_argument(
        "-save_model_path",
        default="p2_best_model.pth",
        type=str,
    )
    parser.add_argument(
        "-train_data_path",
        default="./hw1_data/p2_data/train",
        type=str,
    )
    parser.add_argument(
        "-valid_data_path",
        default="./hw1_data/p2_data/validation",
        type=str,
    )
    parser.add_argument("-model_for_question", default=2, type=int)
    parser.add_argument("-device", default="cuda:0", type=str)
    parser.add_argument("-batch_size", default=4, type=int)
    parser.add_argument("-epochs", default=25, type=int)
    parser.add_argument("-lr", default=7e-5, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
