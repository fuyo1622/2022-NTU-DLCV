# DLCV Fall 2022 Coursework

This repository contains four assignments and the Challenge 1 final project
from the Fall 2022 Deep Learning for Computer Vision course. The source code
has been reformatted and documented with comments explaining data flows,
model structures, and non-obvious operations. The original algorithms,
control flow, and runtime behavior have not been changed.

## Repository Overview

| Directory | Topics | Specification | Report |
| --- | --- | --- | --- |
| [`hw1-fuyo1622`](hw1-fuyo1622/) | Image classification and semantic segmentation | [`hw1_intro.pdf`](hw1-fuyo1622/hw1_intro.pdf) | [`hw1_r11942095.pdf`](hw1-fuyo1622/hw1_r11942095.pdf) |
| [`hw2-fuyo1622`](hw2-fuyo1622/) | GAN, conditional DDPM, and DANN | [`hw2_intro.pptx.pdf`](hw2-fuyo1622/hw2_intro.pptx.pdf) | [`hw2_r11942095.pdf`](hw2-fuyo1622/hw2_r11942095.pdf) |
| [`hw3-fuyo1622`](hw3-fuyo1622/) | CLIP zero-shot classification, image captioning, and attention visualization | [`hw3_intro.pdf`](hw3-fuyo1622/hw3_intro.pdf) | [`hw3_r11942095.pdf`](hw3-fuyo1622/hw3_r11942095.pdf) |
| [`hw4-fuyo1622`](hw4-fuyo1622/) | DVGO novel-view synthesis, BYOL, and Office-Home classification | [`hw4_intro.pdf`](hw4-fuyo1622/hw4_intro.pdf) | [`hw4_r11942095.pdf`](hw4-fuyo1622/hw4_r11942095.pdf) |
| [`final-project-challenge-1-palette`](final-project-challenge-1-palette/) | Talking to Me: multimodal binary classification using audio and face crops | [`DLCV Fall 2022 - Final Project.pdf`](<final-project-challenge-1-palette/DLCV Fall 2022 - Final Project.pdf>) | - |

## Environment

The assignments were originally developed with Python 3.8 and a CUDA-capable
GPU. HW1 through HW4 each provide their own dependency list. Install the
dependencies from inside the corresponding assignment directory:

```bash
python3 -m pip install -r requirements.txt
```

The final project does not include a separate `requirements.txt`. Its main
dependencies are PyTorch, Torchvision, Torchaudio, MoviePy, OpenCV,
face-recognition, PyTorchVideo, scikit-learn, and tqdm.

All relative paths are resolved from the current assignment directory. Change
into the appropriate directory before running any command. The shell scripts
use Bash syntax; Windows users can run them through WSL or Git Bash.

## Data and Model Checkpoints

HW1 through HW3 provide a `get_dataset.sh` script for downloading the course
data. For example:

```bash
cd hw1-fuyo1622
bash get_dataset.sh
```

For the remaining assignments, arrange the data according to the directory
structure documented in the corresponding specification. Pretrained or best
checkpoints can be downloaded with the assignment-specific scripts:

```bash
bash hw1_download.sh
bash hw2_download.sh
bash hw3_download.sh
bash hw4_download.sh
```

Run each download script from its own assignment directory.

## Quick Start

Arguments enclosed in angle brackets are paths that must be replaced.

### HW1

```bash
cd hw1-fuyo1622
bash hw1_download.sh

# Problem 1: image classification -> CSV
bash hw1_1.sh <classification_test_dir> <prediction_csv>

# Problem 2: semantic segmentation -> mask directory
bash hw1_2.sh <segmentation_test_dir> <prediction_dir>
```

The training entry points are `p1_train.py` and `p2_train.py`. The
`p1_plot.py` script projects classification features into two dimensions with
t-SNE or PCA.

### HW2

```bash
cd hw2-fuyo1622
bash hw2_download.sh

# Problem 1: generate face images
bash hw2_1.sh <face_output_dir>

# Problem 2: generate class-conditioned digit images
bash hw2_2.sh <digit_output_dir>

# Problem 3: target-domain digit classification -> CSV
bash hw2_3.sh <target_image_dir> <prediction_csv>
```

Main training and analysis entry points:

- `p1_train_DCGAN.py`: DCGAN baseline.
- `p1_train_SNGAN.py`: spectral-normalized GAN experiment.
- `p2_train.py`: conditional DDPM.
- `p3_train.py`: DANN adaptation from MNIST-M to SVHN or USPS.
- `p3_test.py`: t-SNE visualization of source and target features.

### HW3

```bash
cd hw3-fuyo1622
bash hw3_download.sh

# Problem 1: CLIP zero-shot classification -> CSV
bash hw3_1.sh <image_dir> <id2label_json> <prediction_csv>

# Problem 2: autoregressive image captioning -> JSON
bash hw3_2.sh <image_dir> <caption_json>
```

The `p2.py` script trains the ViT encoder and Transformer decoder. The
`p3.py` script generates cross-attention heatmaps for individual caption
tokens.

### HW4

```bash
cd hw4-fuyo1622
bash hw4_download.sh

# Problem 1: render novel views from test camera poses
bash hw4_1.sh <test_transforms_json> <render_output_dir>

# Problem 2: Office-Home classification -> CSV
bash hw4_2.sh <test_csv> <test_image_dir> <prediction_csv>
```

The entry point for Problem 1 is `run.py`. The core DVGO implementation is in
`lib/`, while scene configurations are stored in `configs/`. For Problem 2,
run `p2_pretrain.py` for BYOL self-supervised pretraining, followed by
`p2_finetune.py` for Office-Home classification.

### Final Project: Talking to Me

The expected data structure is:

```text
train/
├── bbox/
└── seg/
test/
├── bbox/
└── seg/
videos/
└── *.mp4
```

Preprocessing creates segment IDs, filters bounding boxes, crops face images,
extracts audio, produces segment-level WAV files, and assigns routing tags:

```bash
cd final-project-challenge-1-palette
bash preprocess.sh <train_dir> <test_dir> <video_dir>
```

Training and inference entry points:

```bash
bash train.sh
bash inference.sh
```

Inference writes `pred.csv` with the columns `Id,Predicted`.

The original final line of `inference.sh` retains the misspelled filename
`infernece.py`. After the checkpoints have been downloaded, run the existing
inference file directly:

```bash
python3 inference.py
```

## Main Outputs

| Task | Output |
| --- | --- |
| HW1 classification | A `filename,label` CSV file |
| HW1 segmentation | One RGB mask for each input image |
| HW2 GAN | Generated face images in PNG format |
| HW2 DDPM | Generated images named with their digit-class prefix |
| HW2 DANN | An `image_name,label` CSV file |
| HW3 CLIP | A `filename,label` CSV file |
| HW3 captioning | A JSON mapping from image IDs to captions |
| HW4 novel-view synthesis | Rendered PNG images |
| HW4 classification | An `id,filename,label` CSV file |
| Final project | An `Id,Predicted` CSV file |

## Maintenance and Reproducibility Notes

- This is an archived coursework repository. Some paths, CUDA device names,
  and checkpoint names reflect the original experiment environment.
- This cleanup only improves readability and documentation. It does not fix
  existing misspellings, undefined variables, missing imports, or
  environment-specific issues in the original programs.
- Large datasets and most model weights are not tracked. Verify the download
  scripts, data layout, and checkpoint paths before running an experiment.
- The `lib/`, `configs/`, and `tools/` directories in HW4 primarily come from
  the DVGO framework and were not modified during the cleanup.
