#!/bin/sh

# python3 preprocessing.py --video data/videos/zongyi.mp4 --task person-n1 --target_object person --gpu 0 --skip_labeling

# python3 dist_select.py  --video data/videos/zongyi.mp4 --length 141431 --task person-n1 --target_object person \
#                         --chunk_size 640 --diff_delay 10 --diff_thresh 1e-5 --gpu 0 --num_gpus 1

python3 preprocessing.py --video data/videos/zongyi.mp4 --task person-n2 --target_object person --target_object_count 2 --gpu 0 --skip_labeling

# python3 dist_select.py  --video data/videos/zongyi.mp4 --length 141431 \
#                         --task person-n4 --target_object person --target_object_count 4 \
#                         --chunk_size 640 --diff_delay 10 --diff_thresh 1e-5 --gpu 0 --num_gpus 4

python3 preprocessing.py --video data/videos/demo.mp4 --task person-n1 --target_object person --gpu 0

python3 dist_select.py  --video data/videos/demo.mp4 --length 214777 --task person-n1 --target_object person \
                        --chunk_size 640 --diff_delay 10 --diff_thresh 1e-5 --gpu 0 --num_gpus 4
python3 dist_select.py  --video data/videos/demo.mp4 --length 214777 --task person-n1 --target_object person \
                        --chunk_size 640 --diff_delay 10 --diff_thresh 1e-5 --gpu 0 --num_gpus 3
python3 dist_select.py  --video data/videos/demo.mp4 --length 214777 --task person-n1 --target_object person \
                        --chunk_size 640 --diff_delay 10 --diff_thresh 1e-5 --gpu 0 --num_gpus 2
python3 dist_select.py  --video data/videos/demo.mp4 --length 214777 --task person-n1 --target_object person \
                        --chunk_size 640 --diff_delay 10 --diff_thresh 1e-5 --gpu 0 --num_gpus 1

python3 preprocessing.py --video data/videos/demo.mp4 --task person-n2 --target_object person --target_object_count 2 --gpu 0 --skip_labeling
python3 dist_select.py  --video data/videos/demo.mp4 --length 214777 \
                        --task person-n2 --target_object person --target_object_count 2 \
                        --chunk_size 640 --diff_delay 10 --diff_thresh 1e-5 --gpu 0 --num_gpus 4