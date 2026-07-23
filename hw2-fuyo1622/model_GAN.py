# DCGAN and spectral-normalized GAN model definitions for face generation.
import torch.nn as nn
import numpy as np
from torch.nn.utils.parametrizations import spectral_norm
import torch


# Baseline DCGAN generator: project noise through transposed convolutions.
class DC_Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(DC_Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        return self.main(input)


# Baseline DCGAN discriminator for 64-by-64 RGB face images.
class DC_Discriminator(nn.Module):
    def __init__(self):
        super(DC_Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


# Residual upsampling block used by the improved generator.
class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            1,
            padding=1,
        )
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            self.conv2,
        )

        self.bypass_conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            1,
            padding=0,
        )
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        self.bypass = nn.Sequential(
            nn.Upsample(scale_factor=2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# Spectral-normalized residual block with optional spatial downsampling.
class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            1,
            padding=1,
        )
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))

        if stride == 1:
            self.model = nn.Sequential(
                nn.LeakyReLU(0.2),
                spectral_norm(self.conv1),
                nn.LeakyReLU(0.2),
                spectral_norm(self.conv2),
            )
        else:
            self.model = nn.Sequential(
                nn.LeakyReLU(0.2),
                spectral_norm(self.conv1),
                nn.LeakyReLU(0.2),
                spectral_norm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0),
            )

        self.bypass_conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            1,
            padding=0,
        )
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)
        if stride != 1:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0),
            )
        else:
            self.bypass = nn.Sequential(
                spectral_norm(self.bypass_conv),
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# Input discriminator block without the leading activation.
class FirstResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            1,
            padding=1,
        )
        self.bypass_conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            1,
            padding=0,
        )
        nn.init.xavier_uniform_(self.conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, 1.0)

        self.model = nn.Sequential(
            spectral_norm(self.conv1),
            nn.LeakyReLU(0.2),
            spectral_norm(self.conv2),
            nn.AvgPool2d(2),
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            spectral_norm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# Residual generator used by the spectral-normalized GAN.
class SNGAN_Generator(nn.Module):
    def __init__(self, z_dim):
        super(SNGAN_Generator, self).__init__()
        self.z_dim = z_dim

        self.dense = nn.Linear(self.z_dim, 4 * 4 * (64 * 16))
        self.final = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.0)
        nn.init.xavier_uniform_(self.final.weight.data, 1.0)

        self.model = nn.Sequential(
            ResBlockGenerator((64 * 16), (64 * 8)),
            ResBlockGenerator((64 * 8), (64 * 4)),
            ResBlockGenerator((64 * 4), 64 * 2),
            ResBlockGenerator((64 * 2), 64),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            self.final,
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(-1, self.z_dim)
        out = self.dense(z)
        out = out.view(-1, (64 * 16), 4, 4)
        out = self.model(out)
        return out


# Residual discriminator whose linear head is also spectrally normalized.
class SNGAN_Discriminator(nn.Module):
    def __init__(self):
        super(SNGAN_Discriminator, self).__init__()

        self.model = nn.Sequential(
            FirstResBlockDiscriminator(3, 64, stride=2),
            ResBlockDiscriminator(64, 64 * 2, stride=2),
            ResBlockDiscriminator(64 * 2, 64 * 4, stride=2),
            ResBlockDiscriminator(64 * 4, 64 * 8, stride=2),
            ResBlockDiscriminator(64 * 8, 64 * 16, stride=1),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(64 * 16, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.0)
        self.fc = spectral_norm(self.fc)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.model(x)
        features = torch.sum(features, dim=(2, 3))
        out = self.fc(features)
        return out
