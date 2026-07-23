# Problem 1 classification checkpoint
FILE_ID="119ocUOtXlim_SYia1Ha1Xj0g8LRBbXMO"
OUTPUT_NAME='p1_best_model.pth'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt

# Problem 2 segmentation checkpoint
FILE_ID="1JddlN0FXo4pSyBsTLlXISd96Uu9L6f65"
OUTPUT_NAME='p2_best_model.pth'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt
