# Crop tracked people from relevant video frames using filtered boxes.
import cv2
import os
from tqdm import tqdm
import csv
from math import ceil, floor
import argparse


parser = argparse.ArgumentParser(description="Create images for every frame")
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
parser.add_argument(
    "--video_folder",
    type=str,
    help="path to train folder",
    required=True,
)
args = parser.parse_args()

videonames = os.listdir(args.video_folder)

for videoname in tqdm(videonames):
    os.makedirs(os.path.join(args.video_folder, videoname[:-4]))
    bbox = []
    # Select the train or test bounding-box table for this video.
    if os.path.isfile(
        os.path.join(
            args.train_folder,
            "bbox",
            videoname[:-4] + "_newbbox.csv",
        )
    ):
        with open(
            os.path.join(
                args.train_folder,
                "bbox",
                videoname[:-4] + "_newbbox.csv",
            ),
            newline="",
        ) as f:
            rows = csv.reader(f)
            _ = next(rows)
            for row in rows:
                bbox.append(
                    [
                        int(row[0]),
                        int(row[1]),
                        floor(float(row[2])),
                        floor(float(row[3])),
                        ceil(float(row[4])),
                        ceil(float(row[5])),
                    ]
                )
    else:
        with open(
            os.path.join(
                args.test_folder,
                "bbox",
                videoname[:-4] + "_newbbox.csv",
            ),
            newline="",
        ) as f:
            rows = csv.reader(f)
            _ = next(rows)
            for row in rows:
                bbox.append(
                    [
                        int(row[0]),
                        int(row[1]),
                        floor(float(row[2])),
                        floor(float(row[3])),
                        ceil(float(row[4])),
                        ceil(float(row[5])),
                    ]
                )
    bbox.sort(key=lambda x: x[1])
    save_num = 0
    cap = cv2.VideoCapture(os.path.join(args.video_folder, videoname))
    bbox_index = 0

    # Decode sequentially so frame numbers stay aligned with annotations.
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        if bbox_index >= len(bbox):
            pass
        else:
            while bbox[bbox_index][1] == save_num:
                temp = bbox[bbox_index]
                if temp[2] != -1:
                    image = frame[
                        temp[3] : temp[5] + 1,
                        temp[2] : temp[4] + 1,
                        :,
                    ]
                    if image.size == 0:
                        pass
                    else:
                        cv2.imwrite(
                            os.path.join(
                                args.video_folder,
                                videoname[:-4],
                                str(temp[0])
                                + "_"
                                + str(save_num)
                                + ".png",
                            ),
                            image,
                        )
                bbox_index += 1
                if bbox_index >= len(bbox):
                    break
        cv2.waitKey(1000 // 30)
        save_num += 1
    cap.release()
