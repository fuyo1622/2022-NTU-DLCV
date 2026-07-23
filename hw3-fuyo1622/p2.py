# Train the ViT-and-Transformer image-captioning model.
import timm
import torch
import json
from torch.utils.data import Dataset
import torchvision as tv
from typing import Optional, List
from torch import Tensor
import os
from transformers import BertTokenizer
from model import Caption
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import random
import torchvision.transforms.functional as TF
import numpy as np
from tokenizers import Tokenizer

device = "cuda"
# Resolve annotation image IDs to their actual filenames for each split.
with open("train.json", mode="r") as f:
    raw_data = json.load(f)
    train_ann = raw_data["annotations"]
    train_filelist = raw_data["images"]

train_filelist_new = {}
for i in train_filelist:
    train_filelist_new[i["id"]] = i["file_name"]

with open("val.json", mode="r") as f:
    raw_data = json.load(f)
    val_ann = raw_data["annotations"]
    val_filelist = raw_data["images"]

val_filelist_new = {}
for i in val_filelist:
    val_filelist_new[i["id"]] = i["file_name"]


# Randomly rotate training images by one of four right angles.
class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose(
    [
        RandomRotation(),
        tv.transforms.Resize([384, 384]),
        tv.transforms.ColorJitter(
            brightness=[0.5, 1.3],
            contrast=[0.8, 1.5],
            saturation=[0.2, 1.5],
        ),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
)

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


# Keep a padded image tensor together with its valid-pixel mask.
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
    # Pad each image into the fixed 384-by-384 batch canvas.
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


# Pair transformed images with fixed-length tokenized captions and masks.
class ImageDataset(Dataset):
    def __init__(
        self,
        root,
        ann,
        file_list,
        transform=train_transform,
        mode="train",
    ):
        super().__init__()

        self.root = root
        self.transform = transform
        self.ann = ann
        self.mode = mode
        self.file_list = file_list

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        raw_data = self.ann[idx]
        imagename = self.file_list[raw_data["image_id"]]
        image = Image.open(
            os.path.join(self.root, self.mode, imagename)
        ).convert("RGB")
        image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))
        caption_encoded = self.tokenizer.encode(raw_data["caption"])
        caption = caption_encoded.ids
        if len(caption) > 32:
            caption = caption[:32]
        else:
            caption.extend(0 for _ in range(32 - len(caption)))
        cap_mask = (0 == np.array(caption)).astype(bool)
        caption = torch.tensor(caption)
        cap_mask = torch.tensor(cap_mask)
        return (
            image.tensors.squeeze(0),
            image.mask.squeeze(0),
            caption,
            cap_mask,
        )


# Build the pretrained ViT encoder and the trainable caption decoder.
tokenizer = Tokenizer.from_file("caption_tokenizer.json")
nb_tokens = tokenizer.get_vocab_size(False)
train_dataset = ImageDataset(
    "./images",
    train_ann,
    train_filelist_new,
)
val_dataset = ImageDataset(
    "./images",
    val_ann,
    val_filelist_new,
    val_transform,
    mode="val",
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
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-5,
    betas=(0.9, 0.98),
    eps=1e-9,
)


for epoch in range(30):
    # Teacher forcing predicts each next token from the preceding caption.
    model.train()
    criterion.train()
    total_loss = 0
    for images, masks, caps, cap_masks in tqdm(train_loader):
        images = images.to(device)
        caps = caps.to(device)
        cap_masks = cap_masks.to(device)
        outputs = model(
            images,
            caps[:, :-1],
            cap_masks[:, :-1],
        )
        loss = criterion(
            outputs.permute(0, 2, 1),
            caps[:, 1:].long(),
        )
        loss_value = loss.item()
        total_loss += loss_value
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(total_loss / len(train_loader))

    # Evaluate the same shifted-token loss without updating parameters.
    model.eval()
    criterion.eval()
    total_loss = 0
    for images, masks, caps, cap_masks in tqdm(val_loader):
        with torch.no_grad():
            images = images.to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)
            outputs = model(
                images,
                caps[:, :-1],
                cap_masks[:, :-1],
            )
            loss = criterion(
                outputs.permute(0, 2, 1),
                caps[:, 1:].long(),
            )
            loss_value = loss.item()
            total_loss += loss_value
    print(total_loss / len(val_loader))

    torch.save(
        model.state_dict(),
        "./m" + str(epoch) + ".pth",
    )
