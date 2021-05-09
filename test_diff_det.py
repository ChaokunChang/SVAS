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
from utils.parser import *

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

    batch = opt.diff_batch_size
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
