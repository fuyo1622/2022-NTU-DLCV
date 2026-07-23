# Resample full audio tracks and crop train/test clips by frame interval.
import csv
import torchaudio
from tqdm import tqdm
import os


# Create the training audio segments.
seg_names = []
with open("./train.csv", newline="") as f:
    rows = csv.reader(f)
    _ = next(rows)
    for row in rows:
        seg_names.append(row[0])

os.makedirs("./train_seg_audio/", exist_ok=True)
for s in tqdm(seg_names):
    seg_name = s.split("_")
    audio_name = seg_name[0]
    ori_audio, ori_sample_rate = torchaudio.load(
        os.path.join("audios", audio_name + ".wav")
    )
    sample_rate = 16000
    transform = torchaudio.transforms.Resample(ori_sample_rate, sample_rate)

    audio = transform(ori_audio)

    # Convert inclusive 30-fps frame bounds into 16-kHz sample offsets.
    on_set = int(int(seg_name[2]) / 30 * 16000)
    off_set = int((int(seg_name[3]) + 1) / 30 * 16000)

    crop_audio = audio[:, on_set:off_set]
    torchaudio.save(
        os.path.join("train_seg_audio", s + ".wav"),
        crop_audio,
        sample_rate,
    )


# Create the test audio segments.
seg_names = []
with open("./test.csv", newline="") as f:
    rows = csv.reader(f)
    _ = next(rows)
    for row in rows:
        seg_names.append(row[0])

os.makedirs("./test_seg_audio/", exist_ok=True)
for s in tqdm(seg_names):
    seg_name = s.split("_")
    audio_name = seg_name[0]
    ori_audio, ori_sample_rate = torchaudio.load(
        os.path.join("audios", audio_name + ".wav")
    )
    sample_rate = 16000
    transform = torchaudio.transforms.Resample(ori_sample_rate, sample_rate)

    audio = transform(ori_audio)

    # Convert inclusive 30-fps frame bounds into 16-kHz sample offsets.
    on_set = int(int(seg_name[2]) / 30 * 16000)
    off_set = int((int(seg_name[3]) + 1) / 30 * 16000)

    crop_audio = audio[:, on_set:off_set]
    torchaudio.save(
        os.path.join("test_seg_audio", s + ".wav"),
        crop_audio,
        sample_rate,
    )
