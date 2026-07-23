# Problem 1: render novel views with the trained DVGO model.
python3 run.py \
    --config ./configs/nerf/hotdog2.py \
    --render_only \
    --render_test \
    --dump_images \
    --test_json_path $1 \
    --output_folder $2
