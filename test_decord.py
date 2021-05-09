import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import lmdb
import numpy as np
import decord
import os
import itertools
import operator
from decord import VideoReader

import threading
import queue
import sys
import random
import tqdm


class DecordVideoReader():
    def __init__(self, video_file, img_size=(416, 416), gpu=None, num_threads=8, offset=0, is_torch=True):
        self.is_torch = is_torch
        if is_torch:
            decord.bridge.set_bridge('torch')
        if type(img_size) is tuple:
            self.img_size = img_size
        else:
            self.img_size = (img_size, img_size)
        self.offset = offset
        if gpu is None:
            ctx = decord.cpu()
        else:
            ctx = decord.gpu(gpu)
        if type(img_size) == int:
            img_size = (img_size, img_size)
        self._vr = VideoReader(video_file, ctx=ctx, width=img_size[0], height=img_size[1], num_threads=num_threads)

    def __len__(self):
        return len(self._vr)-self.offset

    def __getitem__(self, idx):
        if self.is_torch:
            return self._vr[idx+self.offset].permute(2, 0, 1).contiguous().float().div(255)
        else:
            return self._vr[idx+self.offset].asnumpy()

    def get_batch(self, batch):
        batch = [b+self.offset for b in batch]
        if self.is_torch:
            return self._vr.get_batch(batch).permute(0, 3, 1, 2).contiguous().float().div(255)
        else:
            return self._vr.get_batch(batch).asnumpy()

video = sys.argv[1]
if sys.argv[2] == "None":
    gpu = None
else:
    gpu = int(sys.argv[2])

if len(sys.argv) > 3:
    random_access = int(sys.argv[3])
else:
    random_access = 0
if len(sys.argv) > 4:
    batch = int(sys.argv[4])
else:
    batch = 32

vr = DecordVideoReader(video, img_size=(416, 416), gpu=gpu, offset=0, num_threads=0) # is_torch=False
num_batches = len(vr) // batch
if random_access:
    batches = [[random.randrange(0, len(vr)) for j in range(batch)] for i in range(num_batches)]
else:
    batches = [list(range(i * batch, (i+1) * batch)) for i in range(num_batches)]

scores = None
for batch in tqdm.tqdm(batches, desc="labeling"):
    if len(batch) > 0:
        imgs = vr.get_batch(batch)
        if scores is None:
            scores = imgs
        else:
            scores += imgs
print(type(scores))
a = input("Pause")