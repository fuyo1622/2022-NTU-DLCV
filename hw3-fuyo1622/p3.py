# Generate captions and overlay decoder cross-attention on input images.
import torch
import numpy as np
from transformers import BertTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from model import Caption
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import timm
import torchvision as tv
from tokenizers import Tokenizer
from torchsummary import summary

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(suppress=True)
parser = argparse.ArgumentParser(description="Image Captioning")
parser.add_argument(
    "--folder",
    type=str,
    help="path to images",
    required=True,
)
parser.add_argument(
    "--output",
    type=str,
    help="path to images",
    required=True,
)
args = parser.parse_args()
tokenizer = Tokenizer.from_file("caption_tokenizer.json")
nb_tokens = tokenizer.get_vocab_size(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use the attention-aware caption model checkpoint for visualization.
backbone = timm.create_model(
    "vit_base_patch16_384",
    pretrained=True,
    num_classes=0,
    global_pool="",
).to(device)
model = Caption(
    2048,
    384,
    2048,
    12,
    4,
    32,
    nb_tokens,
    0,
    backbone,
    device,
).to(device)
model.load_state_dict(torch.load("./t9.pth"))
start_token = 2
end_token = 3


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros(
        (1, max_length),
        dtype=torch.long,
    )
    mask_template = torch.ones(
        (1, max_length),
        dtype=torch.bool,
    )

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


# Forward-hook buffers capture the decoder's cross-attention outputs.
features_in_hook = []
features_out_hook = []


def get_attention(self, input, output):
    attn_output = output[0]
    features_in_hook.append(input)
    features_out_hook.append(output[0])


@torch.no_grad()
def evaluate():
    # Generate a caption first, then rerun it while recording attention.
    model.eval()
    caption_len = 0
    for i in range(31):
        predictions, _ = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 3:
            print(caption_len)
            break
        caption_len += 1
        caption[:, i + 1] = predicted_id[0]
        cap_mask[:, i + 1] = False

    model.decoder.layers[3].multihead_attn.register_forward_hook(
        get_attention
    )
    output, attn_map = model(image, caption, cap_mask)

    return (
        caption,
        attn_map[0, : caption_len + 1, 1:],
        caption_len,
    )


def visualize(
    orig_img,
    attn_map,
    caption,
    caption_len,
    filename,
    output_folder,
):
    # Resize each token's patch attention and blend it with the source image.
    attn_map = attn_map.cpu().numpy()
    captions = caption.split(" ")

    results = []
    results.append(orig_img)

    emb_h, emb_w = 24, 24
    for i in range(len(captions) + 1):
        cur_map = attn_map[i]
        new_attn_map = cur_map.reshape(emb_h, emb_w)
        new_attn_map = cv2.resize(
            new_attn_map,
            (orig_w, orig_h),
            interpolation=cv2.INTER_LINEAR,
        )
        final_map = np.stack(
            (new_attn_map, new_attn_map, new_attn_map),
            axis=-1,
        )
        final_map = normalize(
            final_map,
            orig_img.shape[0],
            orig_img.shape[1],
        )
        final_map = final_map.astype(np.uint8)
        final_map = cv2.applyColorMap(final_map, cv2.COLORMAP_JET)
        # cv2.imwrite("results/att-{}.jpg".format(i), final_map)
        result = cv2.addWeighted(orig_img, 0.3, final_map, 0.7, 0)
        cv2.imwrite(
            "results/{}_{}.jpg".format(filename, i),
            result,
        )
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        results.append(result)
    captions.append("<end>")
    captions.insert(0, "<start>")

    assert len(captions) == len(results)

    # Arrange the source image and token overlays in a labeled grid.
    rows, cols = int(np.ceil(len(results) / 5)), 5
    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(16, 9),
    )

    for row in range(rows):
        for col in range(cols):
            ax[row, col].axis("off")

    i = 0
    for row in range(rows):
        if i >= len(results):
            break
        for col in range(cols):
            if i < len(results):
                ax[row, col].imshow(results[i])
                ax[row, col].set_title(captions[i], fontsize=26)
                i += 1
            else:
                break

    name = "{}.png".format(filename)
    plt.savefig(os.path.join(output_folder, name))


val_transform = tv.transforms.Compose(
    [
        tv.transforms.Resize([384, 384]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
)


def normalize(arr, h, w):
    # Scale every heatmap channel independently into the display range.
    new = np.zeros((h, w, 3))
    for j in range(3):
        channel = arr[:, :, j]
        a = 255 / (np.max(channel) - np.min(channel))
        b = -a * (np.min(channel))
        normalized = channel * a + b
        new[:, :, j] = normalized
    return new


if __name__ == "__main__":
    # Process every image independently so each receives its own grid.
    images = [file for file in os.listdir(args.folder)]

    for i in range(len(images)):
        image_path = os.path.join(args.folder, images[i])

        image = Image.open(image_path)
        orig_img = np.array(image)
        orig_w, orig_h = image.size[0], image.size[1]
        image = val_transform(image)
        image = image.unsqueeze(0)
        caption, cap_mask = create_caption_and_mask(
            start_token,
            32,
        )
        image, caption, cap_mask = (
            image.to(device),
            caption.to(device),
            cap_mask.to(device),
        )
        output, attn_map, caption_len = evaluate()

        result = tokenizer.decode(
            output[0].tolist(),
            skip_special_tokens=True,
        )
        result = result.capitalize()
        visualize(
            orig_img=orig_img,
            attn_map=attn_map,
            caption=result,
            caption_len=caption_len,
            filename=images[i][:-4],
            output_folder=args.output,
        )
