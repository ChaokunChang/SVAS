#!/bin/sh

python3 dist_select.py --chunk_size 640 --diff_delay 10 --diff_thresh 1e-5 --gpu 0 --num_gpus 1

python3 dist_select.py --chunk_size 60 --diff_delay 10 --diff_thresh 1e-5 --gpu 0 --num_gpus 1