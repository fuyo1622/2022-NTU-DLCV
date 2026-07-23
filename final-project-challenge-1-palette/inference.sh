# Download the pretrained audio model.
FILE_ID="1SehHGtg9UMRDQR-9YCOE5MAFVDp6O8Eb"
OUTPUT_NAME='./audio_model.pth'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt

# Download the pretrained video model.
FILE_ID="1eVBDmTl4Os47h2CMnzOZe6m3b7dXTZSI"
OUTPUT_NAME='./video_model.pth'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt

# Run inference with the downloaded checkpoints.
python3 infernece.py
