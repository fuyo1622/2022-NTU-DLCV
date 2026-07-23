# DLCV Fall 2022 Coursework

這個 repository 收錄 2022 年秋季「Deep Learning for Computer
Vision」課程的四次作業與 Challenge 1 final project。程式已統一整理
排版並補上資料流程、模型結構及非直觀步驟的註解；演算法、控制流程與
原始執行行為均未更動。

## 專案內容

| 資料夾 | 主題 | 作業說明 | 實驗報告 |
| --- | --- | --- | --- |
| [`hw1-fuyo1622-main`](hw1-fuyo1622-main/) | Image classification、semantic segmentation | [`hw1_intro.pdf`](hw1-fuyo1622-main/hw1_intro.pdf) | [`hw1_r11942095.pdf`](hw1-fuyo1622-main/hw1_r11942095.pdf) |
| [`hw2-fuyo1622-main`](hw2-fuyo1622-main/) | GAN、conditional DDPM、DANN | [`hw2_intro.pptx.pdf`](hw2-fuyo1622-main/hw2_intro.pptx.pdf) | [`hw2_r11942095.pdf`](hw2-fuyo1622-main/hw2_r11942095.pdf) |
| [`hw3-fuyo1622-main`](hw3-fuyo1622-main/) | CLIP zero-shot、image captioning、attention visualization | [`hw3_intro.pdf`](hw3-fuyo1622-main/hw3_intro.pdf) | [`hw3_r11942095.pdf`](hw3-fuyo1622-main/hw3_r11942095.pdf) |
| [`hw4-fuyo1622-main`](hw4-fuyo1622-main/) | DVGO novel-view synthesis、BYOL、Office-Home classification | [`hw4_intro.pdf`](hw4-fuyo1622-main/hw4_intro.pdf) | [`hw4_r11942095.pdf`](hw4-fuyo1622-main/hw4_r11942095.pdf) |
| [`final-project-challenge-1-palette-main`](final-project-challenge-1-palette-main/) | Talking to Me：音訊與人臉影格的多模態二元分類 | [`DLCV Fall 2022 - Final Project.pdf`](<final-project-challenge-1-palette-main/DLCV Fall 2022 - Final Project.pdf>) | - |

## 環境

原作業以 Python 3.8 與 CUDA GPU 為主要執行環境。HW1 至 HW4 各自
提供套件清單，請進入對應資料夾後安裝：

```bash
python3 -m pip install -r requirements.txt
```

Final project 沒有獨立的 `requirements.txt`，主要使用 PyTorch、
Torchvision、Torchaudio、MoviePy、OpenCV、face-recognition、
PyTorchVideo、scikit-learn 與 tqdm。

所有相對路徑都以「目前作業資料夾」為基準，因此執行指令前應先
`cd` 到相應資料夾。Shell 腳本使用 Bash 語法；Windows 使用者可透過
WSL 或 Git Bash 執行。

## 資料與模型權重

HW1 至 HW3 提供 `get_dataset.sh`，可下載課程資料：

```bash
cd hw1-fuyo1622-main
bash get_dataset.sh
```

其他作業請將資料依說明 PDF 所列結構放入對應資料夾。預訓練或最佳
checkpoint 可透過各資料夾的下載腳本取得：

```bash
bash hw1_download.sh
bash hw2_download.sh
bash hw3_download.sh
bash hw4_download.sh
```

每個下載腳本都應在自己的作業資料夾內執行。

## 快速執行

以下參數均為位置參數，尖括號代表需要替換的路徑。

### HW1

```bash
cd hw1-fuyo1622-main
bash hw1_download.sh

# Problem 1: image classification -> CSV
bash hw1_1.sh <classification_test_dir> <prediction_csv>

# Problem 2: semantic segmentation -> mask directory
bash hw1_2.sh <segmentation_test_dir> <prediction_dir>
```

訓練入口為 `p1_train.py` 與 `p2_train.py`；`p1_plot.py` 可將分類特徵
以 t-SNE 或 PCA 投影到二維空間。

### HW2

```bash
cd hw2-fuyo1622-main
bash hw2_download.sh

# Problem 1: generate face images
bash hw2_1.sh <face_output_dir>

# Problem 2: generate class-conditioned digit images
bash hw2_2.sh <digit_output_dir>

# Problem 3: target-domain digit classification -> CSV
bash hw2_3.sh <target_image_dir> <prediction_csv>
```

主要訓練入口：

- `p1_train_DCGAN.py`：DCGAN baseline。
- `p1_train_SNGAN.py`：spectral-normalized GAN 實驗。
- `p2_train.py`：conditional DDPM。
- `p3_train.py`：MNIST-M 到 SVHN／USPS 的 DANN。
- `p3_test.py`：source／target feature 的 t-SNE 視覺化。

### HW3

```bash
cd hw3-fuyo1622-main
bash hw3_download.sh

# Problem 1: CLIP zero-shot classification -> CSV
bash hw3_1.sh <image_dir> <id2label_json> <prediction_csv>

# Problem 2: autoregressive image captioning -> JSON
bash hw3_2.sh <image_dir> <caption_json>
```

`p2.py` 負責訓練 ViT encoder 與 Transformer decoder；`p3.py` 會產生
caption token 對應的 cross-attention heatmap。

### HW4

```bash
cd hw4-fuyo1622-main
bash hw4_download.sh

# Problem 1: render novel views from test camera poses
bash hw4_1.sh <test_transforms_json> <render_output_dir>

# Problem 2: Office-Home classification -> CSV
bash hw4_2.sh <test_csv> <test_image_dir> <prediction_csv>
```

Problem 1 的入口為 `run.py`，核心 DVGO 實作位於 `lib/`，場景設定位於
`configs/`。Problem 2 依序使用 `p2_pretrain.py` 進行 BYOL
self-supervised pretraining，再由 `p2_finetune.py` 訓練 Office-Home
分類器。

### Final Project - Talking to Me

資料結構應包含：

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

前處理會建立 segment IDs、篩選 bounding boxes、裁切人臉影格、擷取
音訊、切割 WAV 片段並建立資料分流標籤：

```bash
cd final-project-challenge-1-palette-main
bash preprocess.sh <train_dir> <test_dir> <video_dir>
```

訓練與推論入口：

```bash
bash train.sh
bash inference.sh
```

推論結果寫入 `pred.csv`，欄位為 `Id,Predicted`。

原始 `inference.sh` 最後一行保留了 `infernece.py` 的檔名拼字；下載完
checkpoint 後，可直接執行實際存在的推論檔：

```bash
python3 inference.py
```

## 主要輸出

| 任務 | 輸出 |
| --- | --- |
| HW1 classification | `filename,label` CSV |
| HW1 segmentation | 每張輸入影像對應的 RGB mask |
| HW2 GAN | 生成的人臉 PNG |
| HW2 DDPM | 以數字類別開頭命名的生成影像 |
| HW2 DANN | `image_name,label` CSV |
| HW3 CLIP | `filename,label` CSV |
| HW3 captioning | image ID 到 caption 的 JSON |
| HW4 novel-view synthesis | 渲染的 PNG 影像 |
| HW4 classification | `id,filename,label` CSV |
| Final project | `Id,Predicted` CSV |

## 維護與重現注意事項

- 這是課程作業封存版本，部分路徑、CUDA 裝置及 checkpoint 名稱採用
  原始實驗環境設定。
- 本次整理只改善可讀性並增加註解，沒有修正原始程式中的拼字、
  未定義變數、缺少 import 或環境相依問題。
- 大型資料集與大部分模型權重不納入版本控制；執行前請先確認下載
  腳本、資料結構與權重路徑。
- HW4 的 `lib/`、`configs/` 與 `tools/` 主要來自 DVGO 相關框架，
  本次未修改其原始實作。
