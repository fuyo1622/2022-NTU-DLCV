# Problem 1: generate face images with the trained GAN.
python3 p1_inference.py \
    -save_model_path 'p1_best_model.pth' \
    -save_folder $1
