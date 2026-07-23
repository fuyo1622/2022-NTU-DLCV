# Domain-Adversarial Neural Network for cross-domain digit recognition.
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


# Shared feature extractor with separate digit and domain prediction heads.
class DANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            Flatten(),
        )
        self.label_classifier = nn.Sequential(
            nn.Linear(3200, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(3200, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x, alpha=0.1, reverse=True):
        features = self.feature_extractor(x)

        # Predict the digit class.
        label = self.label_classifier(features)

        # Predict the input domain.
        if reverse:
            reversed_feature = ReverseLayerF.apply(features, alpha)
            domain = self.domain_classifier(reversed_feature)
        else:
            domain = self.domain_classifier(features)

        return label, domain


# Identity in the forward pass; reverses/scales gradients during backprop.
class ReverseLayerF(Function):
    @staticmethod
    def forward(context, x, alpha):
        context.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(context, grad_output):
        output = grad_output.neg() * context.alpha

        return output, None


# Convert convolutional feature maps into per-sample vectors.
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
