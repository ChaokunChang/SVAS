from numpy.core.numeric import Inf
from numpy.lib.function_base import copy
import ray
from tqdm import tqdm
import argparse
from ray import serve
# import requests
# import random
import time
import torch
from torch import nn
import logging
import numpy as np
import os

from models.difference import SimpleDiff
from models.proxy import tinyresnet18

from dataloader.video_loader import DecordVideoReader

def get_options():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument("--video", type=str, default="data/videos/example.mp4",
                        help="path to the video of interest")
    parser.add_argument("--length", type=int, default=108000,
                        help="specify the length of the video, full length by default")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--img_size", type=int, nargs=2, default=[416, 416])

    # Difference Detector
    parser.add_argument("--diff_model", type=str,
                        help="model of difference detector", default="minus", choices=['minus', 'hist'])
    parser.add_argument("--diff_thresh", type=float,
                        help="threshold of the difference detector", default=30.0)

    # Proxy Model
    parser.add_argument("--proxy_model_ckpt", type=str, default="data/videos/example/checkpoint_final.pt",
                        help="checkpoiny of pre-trained proxy model")
    parser.add_argument("--proxy_batch", type=int, default=64)
    parser.add_argument("--proxy_score_thresh", type=float, default=0.5)
    parser.add_argument("--proxy_score_upper", type=float, default=0.9)
    parser.add_argument("--proxy_score_lower", type=float, default=0.1)

    # Oracle Model
    # A better design is oracle model is yolov5-2, which means a special model for yolov5 which output car only
    parser.add_argument("--oracle_model", type=str, default="yolov5",
                        choices=['yolov5'], help="model of oracle")
    parser.add_argument("--oracle_batch", type=int, default=8)
    parser.add_argument("--oracle_target_label", type=int,
                        default=2, help="2 means car")
    parser.add_argument("--oracle_score_thresh", type=float, default=0.2)

    # Scheduler
    parser.add_argument("--chunk_size", type=int, default=128)

    # Worker
    # pass

    # Other
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--save", default=False,
                        help="save intermediate results", action="store_true")

    opt, _ = parser.parse_known_args()
    opt.video = os.path.abspath(opt.video)

    return opt

def infer(model, test_loader, score_path, device):
    ctx = torch.device(device)

    m = nn.Softmax(dim=1)

    result = []

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Inferencing"):
            images = images.to(ctx)

            feat = model(images)
            probs = m(feat)

            result.append(probs.cpu().numpy())

if __name__ == "__main__":
    opt = get_options()

    videoloader = DecordVideoReader(opt.video, img_size=tuple(opt.img_size), gpu=opt.gpu, offset=opt.offset, is_torch=True)

    diff_detector = SimpleDiff(diff_thresh=opt.diff_thresh)

    batch = opt.proxy_batch
    num_batches = len(videoloader) // batch
    batches = [list(range(i * batch, (i+1) * batch)) for i in range(num_batches)]
    imgs = videoloader.get_batch(batches[0])
    scores = []
    for batch in tqdm(batches, desc="labeling"):
        # if len(batch) > 0:
        #     imgs = videoloader.get_batch(batch)
        #     if scores is None:
        #         scores = imgs
        #     else:
        #         scores += imgs
        probs = diff_detector(imgs,1)
        scores.append(probs)
