# Problem 2: generate conditional digit images with the trained DDPM.
python3 p2_inference.py \
    -save_model_path 'p2_best_model.pth' \
    -save_folder $1
