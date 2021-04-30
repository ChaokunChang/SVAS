import os
import yaml
import random
import argparse
import numpy as np
from os import path as osp
from dataloader.video_loader import DecordVideoReader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config for task')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    train_idx_dir = config['SPLIT']['TRAIN_IDX_DIR']
    val_idx_dir = config['SPLIT']['VAL_IDX_DIR']
    test_idx_dir = config['SPLIT']['TEST_IDX_DIR']
    video_path = config['SOURCE']['VIDEO']
    device = config['DEVICE']

    video_reader = DecordVideoReader(video_path, device=device)
    num_frames = len(video_reader)

    train_frac = config['SPLIT']['TRAIN_FRAC']
    val_frac = config['SPLIT']['VAL_FRAC']

    num_train = int(train_frac * num_frames)
    num_val = int(val_frac * num_frames)

    indices = list(range(num_frames))
    random.shuffle(indices)
    train = indices[:num_train]
    val = indices[num_train:num_train+num_val]
    test = indices[num_train+num_val:]

    train.sort()
    val.sort()
    test.sort()

    os.makedirs(osp.dirname(train_idx_dir), exist_ok=True)
    os.makedirs(osp.dirname(val_idx_dir), exist_ok=True)
    os.makedirs(osp.dirname(test_idx_dir), exist_ok=True)

    np.save(train_idx_dir, train)
    np.save(val_idx_dir, val)
    np.save(test_idx_dir, test)
