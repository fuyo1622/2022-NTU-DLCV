# Problem 2 conditional diffusion checkpoint
FILE_ID="1y6uafJZO73c10DpTRLrI78-3PsNmW3Sx"
OUTPUT_NAME='p2_best_model.pth'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt
