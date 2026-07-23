# Extract one WAV audio track from every source MP4 video.
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import os
import argparse


parser = argparse.ArgumentParser(description="Create ids and labels")
parser.add_argument(
    "--video_folder",
    type=str,
    help="path to train folder",
    required=True,
)
args = parser.parse_args()

# Ignore non-video entries in the supplied video folder.
all_names = os.listdir(args.video_folder)
videonames = []
for names in all_names:
    if names[-4:] == ".mp4":
        videonames.append(names)

os.makedirs("./audios/", exist_ok=True)
for videoname in videonames:
    # Preserve the video stem so later segment IDs can locate the WAV file.
    video = VideoFileClip(os.path.join(args.video_folder, videoname))
    audio = video.audio
    audio.write_audiofile("./audios/" + videoname[:-4] + ".wav")
