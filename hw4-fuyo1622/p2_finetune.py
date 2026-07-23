# Fine-tune a frozen ResNet-50 backbone on Office-Home classification.
import os
import parser
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import OfficeHomeDataset
from model import get_resnet
from tqdm import tqdm, trange


def load_checkpoint(path, model, optimizer):
    state = torch.load(path)
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])


def test(test_loader, test_data, model, device):
    # Return both aggregate accuracy and per-image predicted class indices.
    model.eval()
    labels = []
    correct_count = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            img, label = data[0].to(device), data[1].to(device)
            output = model(img)
            pred = torch.argmax(output, dim=1).detach()
            correct_count += (label == pred).sum()
            pred = pred.cpu().numpy()
            labels.append(pred)
    labels = np.concatenate(labels)
    model.train()
    accuracy = (correct_count / len(test_data)) * 100
    print(
        "Accuracy: {}/{} ({:3f}%)".format(
            correct_count,
            len(test_data),
            accuracy,
        )
    )
    return accuracy, labels


if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    device = "cuda"
    # Dataset construction also establishes the label/index mapping.
    train_dataset = OfficeHomeDataset(
        "./office/train.csv",
        "./office/train",
        mode="train",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=1,
        shuffle=True,
    )
    test_dataset = OfficeHomeDataset(
        "./office/val.csv",
        "./office/val",
        mode="train",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=1,
        shuffle=False,
    )
    # Optimize only parameters left trainable by get_resnet.
    model = get_resnet().to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4,
        weight_decay=1e-5,
    )

    best_acc = 0
    best_epoch = -1
    for epoch in trange(100):
        # Train the classification head for one epoch.
        model.train()
        total_hit = 0
        total_num = 0
        for idx, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            loss = criterion(outputs, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            total_hit += torch.sum(
                torch.argmax(outputs, dim=1) == label
            ).item()
            total_num += image.shape[0]
            if idx % 32 == 0 and idx != 0:
                print(
                    "Epoch: {} [{}/{}] | Loss: {:.4f} | "
                    "Acc: {:.4f} | Best Acc: {:.4f} ({})".format(
                        epoch,
                        idx,
                        len(train_loader),
                        loss.item(),
                        total_hit / total_num,
                        best_acc,
                        best_epoch,
                    )
                )

        # Evaluate after each epoch.
        acc, classes = test(
            test_loader,
            test_dataset,
            model,
            device,
        )

        # Save predictions and a checkpoint whenever validation improves.
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            filenames = test_dataset.filenames
            labels = []
            for c in classes:
                labels.append(test_dataset.class2label[c])
            ids = test_dataset.ids
            df = pd.DataFrame(
                {
                    "id": ids,
                    "filename": filenames,
                    "label": labels,
                }
            )
            df.to_csv("./pred.csv", index=False)
            state = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                state,
                os.path.join(
                    "checkpoints",
                    "{:04d}.pth".format(epoch),
                ),
            )
