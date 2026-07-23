# Train DANN for MNIST-M-to-SVHN or MNIST-M-to-USPS adaptation.
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


def main(args):
    batch_size = args.batch_size
    device = args.device
    source_folder = os.path.join(
        args.train_data_path,
        args.source_domain,
    )
    target_folder = os.path.join(
        args.train_data_path,
        args.target_domain,
    )
    # Pair labeled source batches with target batches for adversarial training.
    source_dataset = ImageDataset(source_folder, "train")
    target_dataset = ImageDataset(target_folder, "val")
    train_target_dataset = ImageDataset(target_folder, "train")
    source_loader = DataLoader(
        source_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    train_target_loader = DataLoader(
        train_target_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_target_loader = DataLoader(
        target_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    model = DANN().to(device)

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs
    best_acc = -1
    save_model_dir = os.path.join(args.save_model_folder)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    # Encode whether domain adaptation was enabled in the checkpoint name.
    if args.no_da:
        save_model_path = (
            args.source_domain
            + "_"
            + args.target_domain
            + "best_model.pth"
        )
    else:
        save_model_path = (
            "da_"
            + args.source_domain
            + "_"
            + args.target_domain
            + "best_model.pth"
        )

    for epoch in range(epochs):
        print("EPOCH:", epoch)
        for i, (src_batch, tar_batch) in enumerate(
            tzip(source_loader, train_target_loader)
        ):
            model.train()
            # Gradually strengthen gradient reversal over the training run.
            p = (
                float(
                    i
                    + epoch
                    * min(
                        len(source_loader),
                        len(train_target_loader),
                    )
                )
                / epochs
                / min(
                    len(source_loader),
                    len(train_target_loader),
                )
            )
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1

            src_img = src_batch[0].to(device)
            src_label = src_batch[1].to(device)
            tar_img = tar_batch[0].to(device)
            tar_label = tar_batch[1].to(device)

            # Source samples supervise both the digit and source-domain heads.
            domain_label = torch.zeros(
                (src_img.size(0), 1),
                dtype=torch.float,
                device=device,
            )
            pred_class, pred_domain = model(src_img, alpha)
            src_class_loss = class_criterion(pred_class, src_label)
            src_domain_loss = domain_criterion(
                pred_domain,
                domain_label,
            )
            # Target samples contribute only domain loss in adaptation mode.
            tar_domain_loss = 0.0
            if not args.no_da:
                domain_labels = torch.ones(
                    (tar_img.size(0), 1),
                    dtype=torch.float,
                    device=device,
                )
                _, pred_domain = model(tar_img, alpha)
                tar_domain_loss = domain_criterion(
                    pred_domain,
                    domain_labels,
                )

            optimizer.zero_grad()
            batch_loss = (
                src_class_loss
                + 1.5 * src_domain_loss
                + 1.5 * tar_domain_loss
            )
            batch_loss.backward()
            optimizer.step()

        # Select checkpoints using target validation classification accuracy.
        model.eval()

        val_loss = 0
        val_acc = 0
        total_batch = 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_target_loader)):
                imgs = batch[0].to(device)
                logits, _ = model(imgs, alpha=0)
                labels = batch[1].to(device)
                batch_loss = class_criterion(logits, labels).item()
                val_loss += batch_loss
                preds = logits.argmax(1)
                acc = torch.sum(preds == labels).item() / labels.size(0)
                val_acc += acc
                total_batch += 1

        val_loss /= total_batch
        val_acc /= total_batch
        print(val_loss, val_acc)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(save_model_dir, save_model_path),
            )

    print(best_acc)


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
    parser.add_argument("-lr", default=0.001, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
