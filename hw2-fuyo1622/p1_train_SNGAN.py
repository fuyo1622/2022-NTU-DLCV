# Configure the spectral-normalized GAN experiment for face generation.
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import os
import torchvision
from face_recog import face_recog
from face_dataset import CelebA_dataset
from model_GAN import SNGAN_Generator, SNGAN_Discriminator


def main(args):
    # Seed the experiment so its latent samples are reproducible.
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

    # Build the residual generator and spectral-normalized discriminator.
    print("SET MODEL")

    discriminator = SNGAN_Discriminator().to(device)
    generator = SNGAN_Generator(z_dim=z_dim).to(device)
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

    print("SET MODEL DONE")

    true_label_value = args.true_label_value
    num_epochs = args.epochs
    fixed_z = Variable(torch.randn(args.num_sample, z_dim).cuda())
    sample_dir = os.path.join(args.sample_folder)
    save_model_dir = os.path.join(args.save_model_folder)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    print("TRAINING")
    print(discriminator)
    print(generator)
    # The original training loop is retained below as a disabled code block.
    '''
	for epoch in range(num_epochs):
		generator.train()
		discriminator.train()
		for i, data in enumerate(tqdm(train_loader)):
			real_imgs = Variable(data.type(torch.cuda.FloatTensor))

			optimizerD.zero_grad()
			optimizerG.zero_grad()
			z = Variable(torch.randn(batch_size,z_dim).cuda())

			gen_imgs = generator(z)
			d_loss = nn.ReLU()(true_label_value - discriminator(real_imgs)).mean() + nn.ReLU()(true_label_value + discriminator(gen_imgs)).mean()
			d_loss.backward()
			optimizerD.step()
			optimizerD.zero_grad()
			optimizerG.zero_grad()
			z = Variable(torch.randn(batch_size,z_dim).cuda())
			gen_imgs = generator(z)		
			g_loss = -discriminator(gen_imgs).mean()

			g_loss.backward()
			optimizerG.step()
			if i % 100 == 0:
				print("G_LOSS: ",g_loss.item(),", D_LOSS: ",d_loss.item())
		
	
		generator.eval()

		samples = (generator(fixed_z).data+1) / 2

		for i, pic in enumerate(tqdm(samples)):
			torchvision.utils.save_image(pic,os.path.join(log_dir, str(i)+'.jpg'))


		face_acc = face_recog(sample_dir)

		print("EPOCH:",epoch,face_acc)
		if face_acc >= 90:

			torch.save(discriminator.state_dict(),os.path.join(save_model_dir,str(face_acc)+"_"+str(epoch+1)+"Dmodel.pth"))
			torch.save(generator.state_dict(),os.path.join(save_model_dir,str(face_acc)+"_"+str(epoch+1)+"Gmodel.pth"))
		if epoch==179:
			torch.save(discriminator.state_dict(),os.path.join(save_model_dir,str(epoch+1)+"Dmodel.pth"))
			torch.save(generator.state_dict(),os.path.join(str(epoch+1)+"Gmodel.pth"))
	print("TRAIN DONE")
	'''

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
