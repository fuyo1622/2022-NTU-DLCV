# Classify Office-Home test images and write the submission CSV.
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from data import OfficeHomeDataset
from model import get_resnet
from tqdm import tqdm, trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero shot")
    parser.add_argument(
        "--test_csv",
        type=str,
        help="path to csv",
        required=True,
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        help="path to images",
        required=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="path to output",
        required=True,
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Test mode reloads the class-label mappings created during training.
    test_dataset = OfficeHomeDataset(
        args.test_csv,
        args.test_data_dir,
        mode="test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=1,
        shuffle=False,
    )
    # Restore the full fine-tuned classifier for prediction.
    model = get_resnet(inference=True).to(device)
    state = torch.load("p2_best.pth")
    model.load_state_dict(state["state_dict"])
    model.eval()

    # Collect numeric predictions in the dataset's original row order.
    classes = []
    with torch.no_grad():
        for idx, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            pred = torch.argmax(output, dim=1).detach()
            pred = pred.cpu().numpy()
            classes.append(pred)

    classes = np.concatenate(classes)
    filenames = test_dataset.filenames
    # Convert class indices back to the assignment's string labels.
    labels = []
    for c in classes:
        labels.append(test_dataset.class2label[c])
    ids = test_dataset.ids
    df = pd.DataFrame(
        {
            "id": ids,
            "filename": filenames,
            "label": labels,
        }
    )
    dir_name = os.path.dirname(args.save_path)
    os.makedirs(dir_name, exist_ok=True)
    df.to_csv(args.save_path, index=False)
