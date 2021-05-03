#!/bin/sh

# GPU sequential access
python3 test_decord.py data/videos/example.mp4 0 0 64

# GPU random access
python3 test_decord.py data/videos/example.mp4 0 1 64

# CPU sequential access
python3 test_decord.py data/videos/example.mp4 None 0 64

# CPU random access
python3 test_decord.py data/videos/example.mp4 None 1 64