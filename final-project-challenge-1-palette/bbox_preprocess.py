# Filter raw face boxes down to frames referenced by train/test segments.
import cv2
import os
import csv
import argparse


parser = argparse.ArgumentParser(description="Create used bbox")
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


# Keep only the bounding boxes used by training segments.
train_segnames = os.listdir(os.path.join(args.train_folder, "seg"))
for segname in tqdm(train_segnames):
    use_frame = []
    use_bbox = []
    with open(
        os.path.join(args.train_folder, "seg", segname),
        newline="",
    ) as f:
        rows = csv.reader(f)
        _ = next(rows)
        for row in rows:
            if int(row[0]) > len(use_frame):
                for i in range(int(row[0]) - len(use_frame)):
                    use_frame.append([])
            else:
                pass
            # Segment end frames are inclusive in the provided annotations.
            to_add = [i for i in range(int(row[1]), int(row[2]) + 1)]
            use_frame[int(row[0]) - 1].extend(to_add)

    with open(
        os.path.join(
            args.train_folder,
            "bbox",
            segname[:-8] + "_bbox.csv",
        ),
        newline="",
    ) as f:
        rows = csv.reader(f)
        use_bbox.append(next(rows))
        for row in rows:
            if int(row[0]) > len(use_frame):
                pass
            else:
                if int(row[1]) in use_frame[int(row[0]) - 1]:
                    use_bbox.append(row)
                else:
                    pass
    with open(
        os.path.join(
            args.train_folder,
            "bbox",
            segname[:-8] + "_newbbox.csv",
        ),
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerows(use_bbox)


# Keep only the bounding boxes used by test segments.
test_segnames = os.listdir(os.path.join(args.test_folder, "seg"))
for segname in tqdm(test_segnames):
    use_frame = []
    use_bbox = []
    with open(
        os.path.join(args.test_folder, "seg", segname),
        newline="",
    ) as f:
        rows = csv.reader(f)
        _ = next(rows)
        for row in rows:
            if int(row[0]) > len(use_frame):
                for i in range(int(row[0]) - len(use_frame)):
                    use_frame.append([])
            else:
                pass
            # Segment end frames are inclusive in the provided annotations.
            to_add = [i for i in range(int(row[1]), int(row[2]) + 1)]
            use_frame[int(row[0]) - 1].extend(to_add)

    with open(
        os.path.join(
            args.test_folder,
            "bbox",
            segname[:-8] + "_bbox.csv",
        ),
        newline="",
    ) as f:
        rows = csv.reader(f)
        use_bbox.append(next(rows))
        for row in rows:
            if int(row[0]) > len(use_frame):
                pass
            else:
                if int(row[1]) in use_frame[int(row[0]) - 1]:
                    use_bbox.append(row)
                else:
                    pass
    with open(
        os.path.join(
            args.test_folder,
            "bbox",
            segname[:-8] + "_newbbox.csv",
        ),
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerows(use_bbox)
