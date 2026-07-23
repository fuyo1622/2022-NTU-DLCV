# Problem 1 DVGO logs and checkpoints
FILE_ID="14xGNe8vLB-6KXg1_rHstKHW7O5St6V2U"
OUTPUT_NAME='./logs.zip'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt
unzip ./logs.zip

# Problem 2 fine-tuned classification checkpoint
FILE_ID="1p-YrF7FGu3Ay2LLlTuCXkwUAIX3Ceytu"
OUTPUT_NAME='./p2_best.pth'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILE_ID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id="$FILE_ID -O $OUTPUT_NAME  && rm -rf /tmp/cookies.txt
