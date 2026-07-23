# Perform CLIP zero-shot classification with text prompts from id2label.
import os
import clip
import torch
import csv
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Zero shot")
parser.add_argument(
    "--folder",
    type=str,
    help="path to images",
    required=True,
)
parser.add_argument(
    "--id2label",
    type=str,
    help="path to id2label",
    required=True,
)
parser.add_argument(
    "--output",
    type=str,
    help="path to output",
    required=True,
)
parser.add_argument(
    "--inference",
    action="store_true",
    default=False,
    help="inference?",
)
args = parser.parse_args()
# Load CLIP once so both datasets share its exact preprocessing pipeline.
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)


# Validation data derives its ground-truth class from the filename prefix.
class ImageDataset(Dataset):
    def __init__(self, root):
        super().__init__()

        self.root = root
        self.imagenames = os.listdir(root)

    def __len__(self):
        return len(self.imagenames)

    def __getitem__(self, idx):
        imagename = self.imagenames[idx]
        image = Image.open(
            os.path.join(self.root, imagename)
        ).convert("RGB")
        return (
            preprocess(image).unsqueeze(0),
            int(imagename.split("_")[0]),
            imagename,
        )


# Test data uses the same preprocessing but has no usable class label.
class TestImageDataset(Dataset):
    def __init__(self, root):
        super().__init__()

        self.root = root
        self.imagenames = os.listdir(root)

    def __len__(self):
        return len(self.imagenames)

    def __getitem__(self, idx):
        imagename = self.imagenames[idx]
        image = Image.open(
            os.path.join(self.root, imagename)
        ).convert("RGB")
        return preprocess(image).unsqueeze(0), 0, imagename


# Turn every class name into the prompt embeddings used for comparison.
with open(args.id2label, mode="r") as f:
    raw_data = json.load(f)
    classes = list(raw_data.values())

text_inputs = torch.cat(
    [clip.tokenize(f"a photo of a {c}") for c in classes]
).to(device)
if not args.inference:
    val_dataset = ImageDataset(args.folder)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
    )
else:
    val_dataset = TestImageDataset(args.folder)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
    )

# Rank normalized image-text similarities and retain the best class ID.
answers = []
for images, labels, imagenames in tqdm(val_loader):
    images = images.to(device)
    for image, label, imagename in zip(
        images,
        labels,
        imagenames,
    ):
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (
            100.0 * image_features @ text_features.T
        ).softmax(dim=-1)
        values, indices = similarity[0].topk(1)
        answers.append([imagename, indices.item()])

# Write evaluator-ready filename and label pairs.
dirname = os.path.dirname(args.output)
if not os.path.isdir(dirname):
    os.makedirs(dirname)
with open(args.output, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    for ans in answers:
        writer.writerow(ans)
