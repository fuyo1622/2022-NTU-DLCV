# Problem 2: run semantic-segmentation inference.
python3 p2_inference.py \
    -model_path 'p2_best_model.pth' \
    -test_data_path $1 \
    -dest_dir $2 \
    -batch_size 2 \
    -device "cuda:0" 
