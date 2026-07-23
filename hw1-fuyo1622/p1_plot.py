# Visualize learned classification features with t-SNE or PCA.
import os
import argparse
import numpy as np
from torch.utils.data import DataLoader

import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm.auto import tqdm
from dataset import ClassficationDataset
from model import tSNE_resnet18


def main(args):
    device = args.device
    valid_dataset = ClassficationDataset(args.valid_data_path, "val")
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = tSNE_resnet18(num_classes=50).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location="cuda"))

    model.eval()

    # Extract the final convolutional representation for every sample.
    predictions = []
    labels = []
    for batch in tqdm(valid_loader):
        img, label, _ = batch
        with torch.no_grad():
            features = model(img.to(device))
        label = label.unsqueeze(1).long()
        for f, l in zip(features, label):
            predictions += [f.cpu().numpy()]
            labels += [l.numpy()]

    x = np.concatenate(predictions).reshape(len(predictions), -1)
    y = np.concatenate(labels)
    colors = np.random.rand(50, 3)

    # Reduce the flattened features to two dimensions for plotting.
    if args.mode == "tSNE":
        tsne = TSNE()
        x_output = tsne.fit_transform(x)
    else:
        pca = PCA(n_components=2)
        x_output = pca.fit_transform(x)

    # Normalize both axes so label annotations fit in a common canvas.
    x_min, x_max = x_output.min(0), x_output.max(0)
    x_norm = (x_output - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    for i in range(x_norm.shape[0]):
        plt.text(
            x_norm[i, 0],
            x_norm[i, 1],
            str(y[i]),
            color=colors[y[i]],
            fontdict={"weight": "bold", "size": 9},
        )
    plt.xticks([])
    plt.yticks([])
    plt.savefig(args.dest_path)


def _parse_args():
    parser = argparse.ArgumentParser(description="Script to plot.")
    parser.add_argument("-valid_data_path", default="val_50", type=str)
    parser.add_argument("-model_path", default="p1_1_model.pth", type=str)
    parser.add_argument("-dest_path", default="output.png", type=str)
    parser.add_argument("-mode", default="tSNE", type=str)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-device", default="cuda:0", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
