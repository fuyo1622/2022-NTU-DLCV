# Train the conditional DDPM and score generated digits during training.
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import csv
import os
import random
from PIL import Image
import torch.optim as optim
from cDDPM_model import DDPM, ContextUnet
from digits_dataset import ImageDataset
import argparse
from digit_classifier import Classifier


def main(args):
    n_epoch = args.epochs
    batch_size = args.batch_size
    n_T = args.n_t
    device = args.device
    n_feat = args.n_feat
    save_model_dir = args.save_model_folder
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    w = 2.0

    # The fixed classifier estimates the quality of generated digit batches.
    test_model = Classifier().to(device)
    ckpt = torch.load(args.test_model_path)
    test_model.load_state_dict(ckpt["state_dict"])

    # Construct the conditional denoiser and its MNIST-M training loader.
    ddpm = DDPM(
        nn_model=ContextUnet(
            in_channels=3,
            n_feat=n_feat,
            n_classes=10,
        ),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1,
    )
    ddpm.to(device)
    train_dataset = ImageDataset(
        os.path.join(args.train_data_path),
        "train",
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = optim.Adam(ddpm.parameters(), lr=args.lr)

    for epoch in range(n_epoch):
        print(f"epoch {epoch}")
        ddpm.train()
        # Decay the learning rate linearly across diffusion-training epochs.
        optimizer.param_groups[0]["lr"] = args.lr * (
            1 - epoch / n_epoch
        )

        pbar = tqdm(train_dataloader)
        loss_ema = None
        for imgs, labels in pbar:
            optimizer.zero_grad()
            imgs = imgs.to(device)
            labels = labels.to(device)
            loss = ddpm(imgs, labels)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optimizer.step()

        # Sample every class and evaluate outputs with the frozen classifier.
        ddpm.eval()
        test_model.eval()
        total_acc = 0
        with torch.no_grad():
            for i in tqdm(range(n_classes)):
                z = torch.randn(10, *(3, 28, 28)).to(device)
                x_gen, _ = ddpm.sample(
                    z,
                    i,
                    10,
                    (3, 28, 28),
                    device,
                    guide_w=w,
                )
                logits = test_model(x_gen)
                _, pred = torch.max(logits, 1)
                label = torch.full(
                    (10,),
                    float(i),
                    dtype=torch.float,
                    device=device,
                )
                acc = (pred == label.to(device)).detach().sum().item()
                total_acc += acc
            print(total_acc / 100)
            if total_acc / 100 >= 0.88:
                torch.save(
                    ddpm.state_dict(),
                    save_dir + f"{w}_model_{ep}.pth",
                )
            total_acc = 0

        if epoch == int(n_epoch - 1):
            torch.save(
                ddpm.state_dict(),
                save_dir + f"model_{ep}.pth",
            )
            print("saved model at " + save_dir + f"model_{ep}.pth")


def _parse_args():
    parser = argparse.ArgumentParser(description="Script to train.")
    parser.add_argument(
        "-save_model_folder",
        default="./digits/cDDPM_model/",
        type=str,
    )
    parser.add_argument(
        "-train_data_path",
        default="./digits/mnistm",
        type=str,
    )
    parser.add_argument(
        "-test_model_path",
        default="./Classifier.pth",
        type=str,
    )
    parser.add_argument("-device", default="cuda", type=str)
    parser.add_argument("-batch_size", default=256, type=int)
    parser.add_argument("-epochs", default=30, type=int)
    parser.add_argument("-n_t", default=400, type=int)
    parser.add_argument("-n_feat", default=256, type=int)
    parser.add_argument("-lr", default=0.0001, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
