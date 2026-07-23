# Visualize how DANN aligns source and target feature distributions.
from digits_dataset import ImageDataset
from DANN_model import DANN
from sklearn.manifold import TSNE
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import argparse
from torchvision import models, transforms
import matplotlib.pyplot as plt


def main(args):
    batch_size = args.batch_size
    device = args.device
    source_folder = "./digits/mnistm"
    source_dataset = ImageDataset(source_folder, "val")
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_folder = "./digits/usps"
    test_dataset = ImageDataset(test_folder, "val")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    # Retain only the trained shared feature extractor for visualization.
    model = DANN().to(device)
    ckpt = torch.load("./digits/da_mnistm_uspsbest_model.pth")
    model.load_state_dict(ckpt)
    model = model.feature_extractor
    plot_tSNE_domains(
        model,
        [source_loader, test_loader],
        ["mnistm", "usps"],
        "./domain1.png",
    )


def plot_tSNE_domains(model, val_loaders, domains, dest_path):
    """ Evaluate on target `val_loaders` with model at `model_path` """

    device = next(model.parameters()).device

    print("Inferencing.")
    model.eval()
    # Assign a domain index to each extracted feature vector.
    features, labels = [], []
    for domain_idx, val_loader in enumerate(val_loaders):
        for i, batch in enumerate(val_loader):
            with torch.no_grad():
                img = batch[0].to(device)
                label = torch.full(
                    size=(img.size(0),),
                    fill_value=domain_idx,
                    dtype=torch.long,
                )
                feature = model(img).cpu()  # [bs, 3200]

            features.append(feature)
            labels.append(label)
            print(
                "Predicting {}/{}".format(i + 1, len(val_loader)),
                end="\r",
            )

    X = torch.cat(features, dim=0)  # features = [N, 3200]
    y = torch.cat(labels, dim=0)  # labels = [N, 1] (digits)

    # Project all domains jointly before plotting their normalized coordinates.
    colors = np.random.rand(2, 3)
    print("Fitting t-SNE" + " " * 10)
    tsne = TSNE(n_components=2, init="pca", random_state=501)
    X_tsne = tsne.fit_transform(X)
    X_norm = (X_tsne - X_tsne.min(0)) / (
        X_tsne.max(0) - X_tsne.min(0)
    )

    plt.figure(figsize=(5, 5))
    for i in range(colors.shape[0]):
        plt.scatter(
            X_norm[y == i, 0],
            X_norm[y == i, 1],
            s=5,
            label=f"{domains[i]}",
        )  # c=colors[i].reshape(1, -1),

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(
        f"t-SNE for {domains[0]}-{domains[1]} case versus domains"
    )
    plt.savefig(dest_path)
    plt.close()


def _parse_args():
    parser = argparse.ArgumentParser(description="Script to train.")
    parser.add_argument("-no_da", default=False, action="store_true")
    parser.add_argument(
        "-save_model_folder",
        default="./digits/DANN_model/",
        type=str,
    )
    parser.add_argument(
        "-train_data_path",
        default="./digits",
        type=str,
    )
    parser.add_argument(
        "-source_domain",
        "-s",
        default="mnistm",
        type=str,
        choices=("mnistm", "svhn", "usps"),
    )
    parser.add_argument(
        "-target_domain",
        "-t",
        default="svhn",
        type=str,
        choices=("svhn", "usps"),
    )
    parser.add_argument("-device", default="cuda", type=str)
    parser.add_argument("-batch_size", default=128, type=int)
    parser.add_argument("-epochs", default=100, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
