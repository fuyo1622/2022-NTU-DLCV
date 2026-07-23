# Run semantic-segmentation inference and save color-coded masks.
import torch
import os
import torchvision
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn
from tqdm.auto import tqdm
import torchvision.transforms as transforms
import argparse
from torch.utils.data import Dataset, DataLoader
from dataset import SegmentationDataset


def main(args):
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    device = args.device
    # The dataset collator batches normalized images and retains sample IDs.
    print("LOAD DATASET")
    test_dataset = SegmentationDataset(args.test_data_path, "val")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
    )
    print("LOAD DATASET DONE")

    # Restore the seven-class DeepLabV3 segmentation model.
    print("SET MODEL")
    model = models.segmentation.deeplabv3_resnet101(num_classes=7).to(
        device
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cuda"))
    model.eval()
    print("SET MODEL DONE")

    # Accumulate class-index masks before converting them back to RGB.
    preditions = {"id": [], "pred": []}
    for batch in tqdm(test_loader):
        img = batch["img"].to(device)
        ids = batch["id"]
        logit = model(img)["out"]
        pred = logit.argmax(1).cpu()
        preditions["id"] += ids
        preditions["pred"] += pred.tolist()

    # Map evaluator class IDs to the assignment's prescribed mask colors.
    class_color = {
        0: [0, 255, 255],
        1: [255, 255, 0],
        2: [255, 0, 255],
        3: [0, 255, 0],
        4: [0, 0, 255],
        5: [255, 255, 255],
        6: [0, 0, 0],
    }
    ids = np.array(preditions["id"])
    preds = np.array(preditions["pred"])
    H, W = preds[0].shape

    print("PREDICTING")
    for ii, (id, pred) in enumerate(zip(ids, preds)):  # pred = [H, W]
        mask_img = np.zeros((H, W, 3))
        for cls, val in class_color.items():
            mask_img[pred == cls] = np.array(val)

        img = Image.fromarray(np.uint8(mask_img))
        img.save(os.path.join(args.dest_dir, f"{id}.png"))
    print("PREDICT DONE")


def _parse_args():
    parser = argparse.ArgumentParser(description="Script to predict.")
    parser.add_argument("-model_path", type=str)
    parser.add_argument("-test_data_path", type=str)
    parser.add_argument("-dest_dir", type=str)
    parser.add_argument("-batch_size", default=4, type=int)
    parser.add_argument("-device", default="cude:0", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
