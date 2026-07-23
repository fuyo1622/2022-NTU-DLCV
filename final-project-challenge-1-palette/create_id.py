# Flatten per-video segment CSV files into train/test segment ID tables.
import csv
import os
import argparse


parser = argparse.ArgumentParser(description="Create ids and labels")
parser.add_argument(
    "--train_folder",
    type=str,
    help="path to train folder",
    required=True,
)
parser.add_argument(
    "--test_folder",
    type=str,
    help="path to test folder",
    required=True,
)
args = parser.parse_args()


# Collect training segment identifiers and labels.
train_seg_names = os.listdir(os.path.join(args.train_folder, "seg"))
total_train_seg_data = [["seg_name", "ttm"]]
for segfile in train_seg_names:
    with open(
        os.path.join(args.train_folder, "seg", segfile),
        newline="",
    ) as f:
        rows = csv.reader(f)
        _ = next(rows)
        for row in rows:
            segname = "_".join([segfile[:-8], row[0], row[1], row[2]])
            total_train_seg_data.append([segname, row[3]])

with open("./train.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(total_train_seg_data)


# Collect test segment identifiers with placeholder labels.
test_seg_names = os.listdir(os.path.join(args.test_folder, "seg"))
total_test_seg_data = [["seg_name", "ttm"]]
for segfile in test_seg_names:
    with open(
        os.path.join(args.test_folder, "seg", segfile),
        newline="",
    ) as f:
        rows = csv.reader(f)
        _ = next(rows)
        for row in rows:
            segname = "_".join([segfile[:-8], row[0], row[1], row[2]])
            total_test_seg_data.append([segname, str(0)])

with open("./test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(total_test_seg_data)
