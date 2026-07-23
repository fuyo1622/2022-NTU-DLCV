# Build segment IDs and filter the bounding-box annotations.
python3 create_id.py --train_folder $1 --test_folder $2
python3 bbox_preprocess.py --train_folder $1 --test_folder $2

# Extract video frames and audio segments, then assign data tags.
python3 video_preprocess.py --train_folder $1 --test_folder $2 --video_folder $3
python3 audio_preprocess.py --video_folder $3
python3 audio_seg.py
python3 create_tag.py
