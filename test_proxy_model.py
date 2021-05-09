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

if __name__ == "__main__":
    opt = get_options()

    videoloader = DecordVideoReader(opt.video, img_size=tuple(opt.img_size), gpu=opt.gpu, offset=opt.offset, is_torch=True)
    proxy = tinyresnet18(num_classes=2, pretrained=False)
    proxy.load_state_dict(torch.load(
        opt.proxy_model_ckpt))
    if opt.gpu is None:
        proxy.to(torch.device('cpu'))
    else:
        proxy.to(torch.device(opt.gpu))
    proxy.eval()

    batch = opt.proxy_batch_size
    num_batches = len(videoloader) // batch
    batches = [list(range(i * batch, (i+1) * batch)) for i in range(num_batches)]
    imgs = videoloader.get_batch(batches[0])
    scores = []
    for batch in tqdm(batches, desc="labeling"):
        feat = proxy(imgs)
        probs = nn.Softmax(dim=1)(feat).detach().cpu().numpy()[:,1]
        scores.append(probs)
