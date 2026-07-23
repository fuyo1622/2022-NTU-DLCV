# Problem 1: run image-classification inference.
python3 p1_inference.py \
    -model_path 'p1_best_model.pth' \
    -test_data_path $1 \
    -dest_file $2 \
    -batch_size 32 \
    -device "cuda:0" 
