#!/bin/sh

# GPU oracle
python3 models/detector.py

# GPU proxy
python3 test_proxy_model.py --video data/videos/example.mp4 --gpu 0 --proxy_batch 64

# CPU proxy
python3 test_proxy_model.py --video data/videos/example.mp4 --proxy_batch 64

# GPU Difference Detector
python3 test_diff_det.py --video data/videos/example.mp4 --gpu 0 --proxy_batch 64

# CPU Difference Detector
python3 test_diff_det.py --video data/videos/example.mp4 --proxy_batch 64
