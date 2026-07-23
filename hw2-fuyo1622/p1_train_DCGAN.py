# Train the baseline DCGAN and evaluate generated faces after each epoch.
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm
from torch.autograd import Variable
import os
import torchvision
import glob
from face_recog import face_recog
from torch.nn import Parameter
from torch.nn.utils.parametrizations import spectral_norm
from face_dataset import CelebA_dataset
from model_GAN import (
    DC_Generator,
    DC_Discriminator,
    SNGAN_Generator,
    SNGAN_Discriminator,
)
import argparse


def main(args):
    # Reuse a fixed seed and latent batch for comparable epoch samples.
    manualSeed = args.fixed_seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    device = args.device
    batch_size = args.batch_size
    z_dim = args.z_dim
    print("LOAD DATASET")

    train_dataset = CelebA_dataset(os.path.join(args.train_data_path))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    print("LOAD DATASET DONE")

    # Configure the adversarial pair and their independent optimizers.
    print("SET MODEL")

    discriminator = DC_Discriminator().to(device)
    generator = DC_Generator(z_dim=z_dim).to(device)
    optimizerD = optim.Adam(
        discriminator.parameters(),
        lr=args.lr_D,
        betas=(0.5, 0.999),
    )
    optimizerG = optim.Adam(
        generator.parameters(),
        lr=args.lr_G,
        betas=(0.5, 0.999),
    )
    criterion = nn.BCELoss()

    print("SET MODEL DONE")

    real_label = 1.0
    fake_label = 0.0

    num_epochs = args.epochs
    fixed_z = torch.randn(
        args.num_sample,
        z_dim,
        1,
        1,
        device=device,
    )
    sample_dir = os.path.join(args.sample_folder)
    save_model_dir = os.path.join(args.save_model_folder)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)

    print("TRAINING")
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        for i, data in enumerate(tqdm(train_loader)):
            # First update the discriminator with real and detached fake data.
            real_imgs = data.to(device)
            optimizerD.zero_grad()
            label = torch.full(
                (batch_size,),
                real_label,
                dtype=torch.float,
                device=device,
            )
            output = discriminator(real_imgs).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            z = torch.randn(batch_size, z_dim, 1, 1, device=device)
            gen_imgs = generator(z)
            label.fill_(fake_label)
            output = discriminator(gen_imgs.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            # Then update the generator to make fake samples appear real.
            optimizerG.zero_grad()
            label.fill_(real_label)
            output = discriminator(gen_imgs).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()
            if i % 100 == 0:
                print(
                    "G_LOSS: ",
                    errG.item(),
                    ", D_LOSS: ",
                    errD.item(),
                )

        # Score a fixed sample set and save checkpoints at chosen milestones.
        generator.eval()

        samples = (generator(fixed_z).data + 1) / 2

        for i, pic in enumerate(tqdm(samples)):
            torchvision.utils.save_image(
                pic,
                os.path.join(log_dir, str(i) + ".jpg"),
            )

        face_acc = face_recog(sample_dir)

        print("EPOCH:", epoch, face_acc)
        if face_acc >= 90:
            torch.save(
                discriminator.state_dict(),
                os.path.join(
                    save_model_dir,
                    str(face_acc)
                    + "_"
                    + str(epoch + 1)
                    + "Dmodel.pth",
                ),
            )
            torch.save(
                generator.state_dict(),
                os.path.join(
                    save_model_dir,
                    str(face_acc)
                    + "_"
                    + str(epoch + 1)
                    + "Gmodel.pth",
                ),
            )
        if epoch == 179:
            torch.save(
                discriminator.state_dict(),
                os.path.join(
                    save_model_dir,
                    str(epoch + 1) + "Dmodel.pth",
                ),
            )
            torch.save(
                generator.state_dict(),
                os.path.join(str(epoch + 1) + "Gmodel.pth"),
            )
    print("TRAIN DONE")


def _parse_args():
    parser = argparse.ArgumentParser(description="Script to train.")
    parser.add_argument(
        "-save_model_folder",
        default="./face/SNGAN_model/",
        type=str,
    )
    parser.add_argument(
        "-train_data_path",
        default="./face/train",
        type=str,
    )
    parser.add_argument(
        "-sample_folder",
        default="./face/sample",
        type=str,
    )
    parser.add_argument("-device", default="cuda", type=str)
    parser.add_argument("-batch_size", default=64, type=int)
    parser.add_argument("-epochs", default=180, type=int)
    parser.add_argument("-z_dim", default=100, type=int)
    parser.add_argument("-num_sample", default=100, type=int)
    parser.add_argument("-fixed_seed", default=911, type=int)
    parser.add_argument("-lr_G", default=0.0001, type=float)
    parser.add_argument("-lr_D", default=0.0004, type=float)
    parser.add_argument("-true_label_value", default=0.9, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
