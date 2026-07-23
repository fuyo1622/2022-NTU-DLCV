# Generate a reproducible set of face images from a trained GAN checkpoint.
import argparse
import os
import random
import torch
import torchvision

from tqdm import tqdm
from torch.autograd import Variable
from model_GAN import SNGAN_Generator, DC_Generator


def main(args):
    # Fix every random source used to construct the latent vectors.
    manualSeed = args.fixed_seed
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    device = args.device
    z_dim = args.z_dim

    # Restore the improved SNGAN generator used for submission.
    print("LOAD MODEL")
    generator = SNGAN_Generator(z_dim=z_dim).to(device)
    ckpt = torch.load(args.save_model_path)
    generator.load_state_dict(ckpt)
    print("LOAD MODEL DONE")

    # Generate one output image for each fixed latent vector.
    fixed_z = Variable(torch.randn(args.num_output, z_dim)).to(device)
    save_dir = os.path.join(args.save_folder)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    generator.eval()
    for i, z in enumerate(tqdm(fixed_z)):
        pic = (generator(z).data + 1) / 2
        torchvision.utils.save_image(
            pic,
            os.path.join(save_dir, str(i) + ".png"),
        )


def _parse_args():
    parser = argparse.ArgumentParser(description="Script to train.")
    parser.add_argument(
        "-save_model_path",
        default="./p1_best_model.pth",
        type=str,
    )
    parser.add_argument("-save_folder", default="./face/output", type=str)
    parser.add_argument("-device", default="cuda", type=str)
    parser.add_argument("-z_dim", default=100, type=int)
    parser.add_argument("-num_output", default=1000, type=int)
    parser.add_argument("-fixed_seed", default=1121, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
