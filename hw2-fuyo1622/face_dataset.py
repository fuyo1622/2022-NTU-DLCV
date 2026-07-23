# CelebA image loader and normalization used by both GAN variants.
import torchvision.transforms as transforms
import glob
import os
from PIL import Image
from torch.utils.data import Dataset


# Return normalized RGB face images without labels.
class CelebA_dataset(Dataset):
    def __init__(self, root):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5, 0.5, 0.5),
                    (0.5, 0.5, 0.5),
                ),
            ]
        )
        self.filenames_list = glob.glob(os.path.join(root, "*"))

    def __getitem__(self, idx):
        filename = self.filenames_list[idx]
        img = Image.open(filename).convert("RGB")
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.filenames_list)
