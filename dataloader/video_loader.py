import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import lmdb
import threading
import queue
import math
import sys
import random
import tqdm
import numpy as np
import decord
import os
import itertools
import operator
from decord import VideoReader



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


class VideoLoader:
    def __init__(self, video_reader:DecordVideoReader, label_reader, indices, batch_size, device, shuffle, enable_cache):
        self.vr = video_reader
        self.lr = label_reader
        self.indices = indices
        self.batch_size = batch_size
        self.queue = queue.Queue(64)
        self.cur = 0
        self.num_samples = len(indices)
        self.len = math.ceil(self.num_samples / batch_size)
        self.device = device
        self.shuffle = shuffle

        # keep samples in memory to eliminate I/O bottleneck
        self.enable_cache = enable_cache
        self.cached = False
        if self.enable_cache:
            self.cache_images = torch.empty(self.num_samples, 3, *video_reader.img_size)
            self.cache_labels = torch.empty(self.num_samples, dtype=torch.int64)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.cur = 0

        # shuffle dataset
        if self.shuffle:
            np.random.shuffle(self.indices)

        if (not self.enable_cache) or (not self.cached):
            threading.Thread(
                target=self.loader_thread,
                args=(
                    self.vr,
                    self.lr,
                    self.indices,
                    self.num_samples,
                    self.batch_size,
                    self.queue
                )
            ).start()
        return self

    def __next__(self):
        if self.cur >= self.len:
            if self.enable_cache:
                self.cached = True
            raise StopIteration

        start = self.cur*self.batch_size
        end = (self.cur+1)*self.batch_size
        if not self.cached:
            images, labels = self.queue.get(block=True)
            if self.enable_cache:
                self.cache_images[start:end] = images
                self.cache_labels[start:end] = labels
        else:
            images = self.cache_images[start:end]
            labels = self.cache_labels[start:end]

        self.cur += 1
        images = images.to(self.device)
        labels = labels.to(self.device)
        return images, labels

    def loader_thread(self, video_reader, label_reader, indices, num_samples, batch_size, queue):
        # set bridge in new thread
        decord.bridge.set_bridge('torch')

        for idx in range(0, num_samples, batch_size):
            start = idx
            end = idx + batch_size
            batch_idx = indices[start:end]
            images = video_reader.get_batch(batch_idx)
            labels = label_reader.get_batch(batch_idx)

            labels = torch.from_numpy(labels).long()
            queue.put((images, labels), block=True)

