# Sample class-conditioned MNIST-M images from a trained DDPM.
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
import torchvision


def main(args):
    # Fix random generators so the submission images remain reproducible.
    manualSeed = args.fixed_seed
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    n_T = args.n_t
    device = args.device
    n_feat = args.n_feat

    # Recreate the denoising U-Net before loading its checkpoint.
    print("LOAD MODEL")
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
    ckpt = torch.load(args.save_model_path)
    ddpm.load_state_dict(ckpt)
    print("LOAD MODEL DONE")

    save_dir = os.path.join(args.save_folder)
    num_of_classes = args.num_output // 10
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Produce an equal number of samples for each digit class.
    ddpm.eval()
    with torch.no_grad():
        fixed_z = torch.randn(
            num_of_classes,
            *(3, 28, 28),
        ).to(device)
        for i in tqdm(range(10)):
            x_gen, _ = ddpm.sample(
                fixed_z,
                i,
                num_of_classes,
                (3, 28, 28),
                device,
                guide_w=2.0,
            )

            for j, img in enumerate(x_gen):
                img = img.data / 2 + 0.5
                torchvision.utils.save_image(
                    img,
                    os.path.join(
                        save_dir,
                        str(i) + "_" + "%03d" % j + ".png",
                    ),
                )


def _parse_args():
    parser = argparse.ArgumentParser(description="Script to train.")
    parser.add_argument(
        "-save_model_path",
        default="./p2_best_model.pth",
        type=str,
    )
    parser.add_argument(
        "-save_folder",
        default="./digits/mnistm/output",
        type=str,
    )
    parser.add_argument("-device", default="cuda", type=str)
    parser.add_argument("-num_output", default=1000, type=int)
    parser.add_argument("-n_t", default=400, type=int)
    parser.add_argument("-n_feat", default=256, type=int)
    parser.add_argument("-fixed_seed", default=911, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
