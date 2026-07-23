# Train and validate the HW1 image-classification models.
import torch
import os
import torchvision
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import argparse
from model import resnet50, resnet18
from torch.utils.data import Dataset, DataLoader
from dataset import ClassficationDataset


def main(args):
    device = args.device

    # Build separate shuffled training and deterministic validation loaders.
    print("LOAD DATASET")

    train_dataset = ClassficationDataset(args.train_data_path, "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataset = ClassficationDataset(args.valid_data_path, "val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("LOAD DATASET DONE")

    # Select the from-scratch ResNet-18 or the ResNet-50 alternative.
    print("SET MODEL")
    num_classes = 50
    if args.model_for_question == 1:
        model = resnet18(num_classes).to(device)
    else:
        model = resnet50(
            num_classes,
            pre_trained=args.pre_trained,
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )

    n_epochs = args.epochs
    best_acc = -1
    best_state_dict = None
    save_path = args.save_model_path
    print("SET MODEL DONE")

    print("TRAINING")
    for epoch in range(n_epochs):
        # Update model weights on the training split.
        model.train()
        train_loss = 0
        train_acc = 0
        for batch in tqdm(train_loader):
            img, label, _ = batch
            logit = model(img.to(device))
            loss = criterion(logit, label.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (logit.argmax(dim=-1) == label.to(device)).float().mean()
            train_loss += loss.item()
            train_acc += acc

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print(
            f"[ TRAIN | {epoch + 1:03d}/{n_epochs:03d} ] "
            f"LOSS = {train_loss:.5f}, ACC = {train_acc:.5f}"
        )

        # Measure validation accuracy without gradient tracking.
        model.eval()
        valid_loss = 0
        valid_acc = 0
        for batch in tqdm(val_loader):
            img, label, _ = batch
            with torch.no_grad():
                logit = model(img.to(device))
            loss = criterion(logit, label.to(device))
            acc = (logit.argmax(dim=-1) == label.to(device)).float().mean()
            valid_loss += loss.item()
            valid_acc += acc

        valid_loss /= len(val_loader)
        valid_acc /= len(val_loader)
        print(
            f"[ VALID | {epoch + 1:03d}/{n_epochs:03d} ] "
            f"LOSS = {valid_loss:.5f}, ACC = {valid_acc:.5f}"
        )

        # Preserve the best validation checkpoint and report milestones.
        if valid_acc >= best_acc:
            best_acc = valid_acc
            best_state_dict = model.state_dict()
            torch.save(best_state_dict, save_path)
            print(
                "SAVE MODEL FOR EPOCH:",
                epoch + 1,
                "ACC=",
                best_acc.item(),
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
        default="p1_best_model.pth",
        type=str,
    )
    parser.add_argument("-train_data_path", default="train_50", type=str)
    parser.add_argument("-valid_data_path", default="val_50", type=str)
    parser.add_argument("-model_for_question", default=1, type=int)
    parser.add_argument("-pre_trained", default=True, type=bool)
    parser.add_argument("-device", default="cuda:0", type=str)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-epochs", default=100, type=int)
    parser.add_argument("-lr", default=0.01, type=float)
    parser.add_argument("-momentum", default=0.9, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
