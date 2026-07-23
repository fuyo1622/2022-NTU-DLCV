# Run HW1 image-classification inference and write the submission CSV.
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
from model import resnet50
from torch.utils.data import Dataset, DataLoader
from dataset import ClassficationDataset


def main(args):
    device = args.device
    if not os.path.exists(os.path.dirname(args.dest_file)):
        os.makedirs(os.path.dirname(args.dest_file))

    # Load test images without labels or random augmentation.
    print("LOAD DATASET")
    test_dataset = ClassficationDataset(args.test_data_path, "test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("LOAD DATASET DONE")
    # Restore the trained classifier before iterating over test batches.
    print("SET MODEL")
    model = resnet50(50, pre_trained=False).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location="cuda"))
    model.eval()
    print("SET MODEL DONE")

    # Keep filenames aligned with their predicted class indices.
    predictions = {"filenames": [], "preds": []}
    print("PREDICTING")
    for batch in tqdm(test_loader):
        img, file_name = batch
        with torch.no_grad():
            logit = model(img.to(device))
        batch_pred = logit.argmax(dim=-1)
        predictions["filenames"] += file_name
        predictions["preds"] += batch_pred.tolist()

    # Serialize predictions in the exact format required by the evaluator.
    pred_str = "filename,label\n"
    for filename, pred in zip(
        predictions["filenames"],
        predictions["preds"],
    ):
        filename = filename.split("/")[-1]
        pred_str += "{},{}\n".format(filename, pred)

    with open(os.path.join(args.dest_file), "w") as f:
        f.write(pred_str)

    print("PREDICT DONE")


def _parse_args():
    parser = argparse.ArgumentParser(description="Script to inference.")
    parser.add_argument("-model_path", type=str)
    parser.add_argument("-test_data_path", type=str)
    parser.add_argument("-dest_file", type=str)
    parser.add_argument("-batch_size", default=32, type=int)
    parser.add_argument("-device", default="cude:0", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args)
