# Autoregressively generate captions for a folder of input images.
import timm
import torch
import json
from torch.utils.data import Dataset
import torchvision as tv
from typing import Optional, List
from torch import Tensor
import os
from model import Caption
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np
from tokenizers import Tokenizer
import argparse

parser = argparse.ArgumentParser(description="Zero shot")
parser.add_argument(
    "--folder",
    type=str,
    help="path to images",
    required=True,
)
parser.add_argument(
    "--output",
    type=str,
    help="path to output",
    required=True,
)
args = parser.parse_args()

val_transform = tv.transforms.Compose(
    [
        tv.transforms.Resize([384, 384]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
)


# Keep image tensors and padding masks in the same container as training.
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # Pad images to the encoder's fixed 384-by-384 input shape.
    if tensor_list[0].ndim == 3:
        max_size = [3, 384, 384]
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(
            batch_shape,
            dtype=dtype,
            device=device,
        )
        mask = torch.ones(
            (b, h, w),
            dtype=torch.bool,
            device=device,
        )
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[
                : img.shape[0],
                : img.shape[1],
                : img.shape[2],
            ].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# Preserve each filename for the final JSON key.
class ImageDataset(Dataset):
    def __init__(self, root, transform=val_transform):
        super().__init__()

        self.root = root
        self.transform = transform
        self.file_list = os.listdir(root)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        imagename = self.file_list[idx]
        image = Image.open(
            os.path.join(self.root, imagename)
        ).convert("RGB")
        image = self.transform(image)

        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        return image.tensors.squeeze(0), imagename


# Recreate the training architecture before restoring its checkpoint.
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_file("caption_tokenizer.json")
nb_tokens = tokenizer.get_vocab_size(False)
val_dataset = ImageDataset(args.folder, val_transform)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
)
backbone = timm.create_model(
    "vit_base_patch16_384",
    pretrained=True,
    num_classes=0,
    global_pool="",
).to(device)
model = Caption(
    2048,
    384,
    2048,
    12,
    4,
    32,
    nb_tokens,
    0,
    backbone,
    device,
).to(device)
model.load_state_dict(torch.load("./p2_best.pth"))
model.eval()
start_token = 2
end_token = 3
ans = {}


def create_caption_and_mask(start_token, max_length, batch_size):
    # Initialize an empty caption with only the start token unmasked.
    caption_template = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
    )
    mask_template = torch.ones(
        (batch_size, max_length),
        dtype=torch.bool,
    )

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


for images, imagenames in tqdm(val_loader):
    # Decode one token at a time until EOS or the maximum length.
    with torch.no_grad():
        images = images.to(device)
        captions, cap_masks = create_caption_and_mask(
            start_token,
            32,
            1,
        )
        captions = captions.to(device)
        cap_masks = cap_masks.to(device)
        for i in range(31):
            predictions = model(images, captions, cap_masks)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)
            if predicted_id[0] == 3:
                break
            captions[:, i + 1] = predicted_id[0]
            cap_masks[:, i + 1] = False
    for i in captions:
        ans[imagenames[0][:-4]] = tokenizer.decode(
            i.tolist(),
            skip_special_tokens=True,
        )

# Store captions by extension-free image ID as required by evaluation.
dirname = os.path.dirname(args.output)
if not os.path.isdir(dirname):
    os.makedirs(dirname)

with open(args.output, "w") as f:
    json.dump(ans, f, indent=4)
