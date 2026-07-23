# Self-supervise a ResNet-50 backbone on Mini-ImageNet with BYOL.
import torch
from byol_pytorch import BYOL
from torchvision import models, transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os
from PIL import Image

# BYOL treats the ResNet average-pooling output as its representation.
resnet = models.resnet50(pretrained=False).to("cuda")

learner = BYOL(
    resnet,
    image_size=128,
    hidden_layer="avgpool",
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
IMAGE_EXTS = [".jpg", ".png", ".jpeg"]


def expand_greyscale(t):
    return t.expand(3, -1, -1)


# Recursively collect supported image files for self-supervised training.
class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f"{folder}").glob("**/*"):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f"{len(self.paths)} images found")

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Lambda(expand_greyscale),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert("RGB")
        return self.transform(img)


# Optimize online BYOL parameters and update the target moving average.
train_dataset = ImagesDataset("./mini/train", 128)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=1,
    shuffle=True,
)
for epoch in range(1000):
    print(epoch)
    for images in tqdm(train_loader):
        images = images.to("cuda")
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()

torch.save(resnet.state_dict(), "./improved-net.pt")
