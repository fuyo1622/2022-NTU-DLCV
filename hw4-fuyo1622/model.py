# ResNet-50 classifier construction for the Office-Home downstream task.
import torch
import torch.nn as nn
from torchvision import models


def get_resnet(inference=False):
    # Replace the ImageNet head with a 65-class multilayer classifier.
    resnet = models.resnet50(pretrained=False)
    resnet.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(256, 65),
    )
    # During fine-tuning, load the supplied backbone and train only the head.
    if not inference:
        resnet.load_state_dict(
            torch.load("./pretrain_model_SL.pt"),
            strict=False,
        )
        for name, param in resnet.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    return resnet
