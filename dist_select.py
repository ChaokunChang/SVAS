from numpy.core.numeric import Inf
from numpy.lib.function_base import copy
import ray
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
import json

from dataloader.video_loader import DecordVideoReader
from dataloader.label_loader import BinaryLabelReader
from models.detector import YOLOv5
from models.difference import SimpleDiff
from models.proxy import tinyresnet18, ProxyTinyResNet
from models.yolo_utils import get_obj_id
from utils.parser import *

ray.init()
serve.start()

@ray.remote(num_gpus=1)
class NoScopeService:
    def __init__(self, sid, opt):
        self.sid = sid
        self.init_state()

        self.videoloader = DecordVideoReader(opt.video, img_size=tuple(opt.img_size), gpu=opt.gpu, offset=opt.offset, is_torch=True)

        self.diff_detector = SimpleDiff(diff_thresh=opt.diff_thresh, delay=opt.diff_delay)

        self.c_low = opt.proxy_score_lower
        self.c_high = opt.proxy_score_upper
        self.proxy = ProxyTinyResNet(num_classes=2, device=opt.gpu, model_ckpt=opt.proxy_model_ckpt)

        self.oracle_batch_size = opt.oracle_batch_size
        self.oracle_type = opt.oracle_model
        self.target_object = opt.target_object
        self.target_object_id = get_obj_id(self.target_object)
        self.target_object_count = opt.target_object_count
        self.target_object_thresh = opt.target_object_thresh
        self.oracle = YOLOv5(model_type=self.oracle_type, thr=self.target_object_thresh,
                             long_side=opt.img_size[0], device=opt.gpu, fp16=False)

        # import logging
        # logging.basicConfig(level=logging.DEBUG, filename=f'./server-{sid}-{opt.diff_thresh}-{opt.chunk_size}-{opt.img_size[0]}.log')
        # self.logger = logging.getLogger(__name__)

    def __call__(self, request):
        offset = request.query_params["offset"]
        length = request.query_params["length"]
        targets, _ = self.run(offset=offset, length=length)
        return {"sid": self.sid, "results": targets}

    def get_mask(self, values, masker):
        if isinstance(values, list):
            return masker(np.array(values))
        else:
            return masker(values)

    def init_state(self, chunk_ids=None):
        self.info = {'sid': self.sid}
        self.meta_store = {}
        if chunk_ids is not None:
            self.meta_store['chunk'] = chunk_ids
            self.meta_store['candidates'] = np.array(list(range(len(chunk_ids))))

    def stage1(self, frames):
        candidates = self.meta_store['candidates']

        s1_time = time.time()

        distance_scores = self.diff_detector(frames)
        # print("DEBUG1: ", distance_scores)
        diff_mask = self.get_mask(distance_scores, lambda x: x >= 0.)
        # print("DEBUG2: ", diff_mask, np.sum(diff_mask))
        frames = frames[diff_mask]
        candidates = candidates[diff_mask]

        e1_time = time.time()

        self.meta_store['diff_mask'] = diff_mask
        self.meta_store['candidates'] = candidates
        self.info['cost-1'] = e1_time - s1_time
        self.info['size-1'] = len(diff_mask)
        return frames

    def stage2(self, frames):
        candidates = self.meta_store['candidates']

        s2_time = time.time()

        proxy_scores = self.proxy(frames)
        # print("DEBUG-stage2-score: ", proxy_scores)
        proxy_mask = self.get_mask(
            proxy_scores, lambda x: np.logical_and(x > self.c_low, x < self.c_high))
        frames = frames[proxy_mask]

        early_candidate_mask = self.get_mask(proxy_scores, lambda x: x >= self.c_high)
        candidates_from_proxy = candidates[early_candidate_mask]
        candidates = candidates[proxy_mask]
        
        e2_time = time.time()

        self.meta_store['candidates_from_proxy'] = candidates_from_proxy
        self.meta_store['candidates'] = candidates
        self.info['cost-2'] = e2_time - s2_time
        self.info['size-2'] = len(proxy_mask)
        return frames

    def stage3(self, frames):
        candidates = self.meta_store['candidates']

        s3_time = time.time()
        labels = []
        batches = torch.split(frames, self.oracle_batch_size)
        for batch in batches:
            if len(batch) == 0:
                break
            b_labels = self.oracle.infer4object(batch, self.target_object_id, self.target_object_count)
            labels.extend(b_labels)
        oracle_mask = self.get_mask(labels, lambda x: x >= 0.5)
        frames = frames[oracle_mask]
        candidates = candidates[oracle_mask]

        e3_time = time.time()

        self.meta_store['candidates'] = candidates
        self.info['cost-3'] = e3_time - s3_time
        self.info['size-3'] = len(oracle_mask)
        return frames

    def run(self, offset, length):
        targets = []
        start_time = time.time()
        
        batch_fids = list(
            range(offset, min(offset+length, len(self.videoloader))))
        self.init_state(chunk_ids=batch_fids)

        # Load frames from video reader according to ids from request
        frames = self.videoloader.get_batch(batch_fids) # return ndarray
        frames = self.stage1(frames)
        frames = self.stage2(frames)
        frames = self.stage3(frames)

        candidates = self.meta_store['candidates']
        candidates_from_proxy = self.meta_store['candidates_from_proxy']
        diff_mask = self.meta_store['diff_mask']
        candidates = np.concatenate((candidates, candidates_from_proxy), axis=0)
        for i in candidates:
            cur = i
            targets.append(batch_fids[cur])
            cur += 1
            while cur < len(batch_fids) and diff_mask[cur] == 0:
                targets.append(batch_fids[cur])
                cur += 1

        end_time = time.time()

        self.info['cost'] = end_time - start_time
        self.info['size'] = len(targets)
        # self.logger.debug(f'{self.info}')

        return targets, self.info

def evaluate(selected_ids, opt):
    # video_reader = DecordVideoReader(
    #     opt.video, gpu = opt.gpu, img_size=tuple(opt.img_size), offset=opt.offset)
    
    gt_label_path = get_label_path(opt)
    target_obj_id = get_obj_id(opt.target_object)
    label_reader = BinaryLabelReader(gt_label_path, opt.offset, target_obj_id, opt.target_object_thresh, opt.target_object_count)

    # train_loader=VideoLoader(video_reader, label_reader, train_idxs,
    #                          batch_size, device, shuffle=True, enable_cache=True)

    gt_labels = label_reader.get_batch(selected_ids)

    gt_labels = np.array(gt_labels)
    tp = (gt_labels == 1).sum()

    precision = tp / len(gt_labels)
    print('Summary:')
    # print(f'positive: {num_pos}')
    # print(f'negative: {num_neg}')
    # print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    # print(f'Recall: {recall}')
    return precision

if __name__ == "__main__":
    opt = get_select_options()
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    num_servers = opt.num_gpus
    servers = []
    for i in range(num_servers):
        servers.append(NoScopeService.remote(sid=i, opt=opt))

    # video_reader = DecordVideoReader(
    #     opt.video, gpu = opt.gpu, img_size=tuple(opt.img_size), offset=opt.offset)
    # video_length = get_video_length(video_reader, opt)
    # del video_reader
    
    video_length = opt.length
    chunk_size = opt.chunk_size
    num_iters = video_length // chunk_size
    niter = 0
    task_queue = []

    results = []
    infos = []

    start_time = time.time()
    while niter < len(servers):
        # logger.debug(f'***** iter-{niter}/{num_iters} with sid={niter} *****')
        offset = niter * chunk_size
        task_queue.append(servers[niter].run.remote(offset, chunk_size))
        niter += 1

    while niter < num_iters:
        # wait will block until the first task finished.
        done, task_queue = ray.wait(task_queue)

        selected_ids, info = ray.get(done)[0]
        results += selected_ids
        infos.append(info)
        logger.debug(f'Iter-{niter-num_servers}/{num_iters} Got {len(selected_ids)} items from server-{info["sid"]}, {info}')
        free_server = servers[info['sid']]
        task_queue.append(free_server.run.remote(niter*chunk_size, chunk_size))
        niter += 1

    while len(task_queue) > 0:
        done, task_queue = ray.wait(task_queue)
        selected_ids, info = ray.get(done)[0]
        results += selected_ids
        infos.append(info)
        logger.debug(f'Iter-{niter-num_servers}/{num_iters} Got {len(selected_ids)} items from server-{info["sid"]}, {info}')
        niter += 1

    end_time = time.time()

    acc = evaluate(results, opt)

    logger.info(
        f'Job Finished, {len(results)} out of {video_length} frames were selected with cost of {end_time - start_time}s. The precision is {acc}')
    
    np.save(get_tmp_results_path(opt), np.array(results))
    with open(get_tmp_infos_path(opt), 'w') as fp:
        json.dump(infos, fp)

# Exp
# GPU num | chunk size | size(results) | acc | time cost
# 1xGPU | 64-chunk | 
# 2xGPU | 64-chunk | 
# 3xGPU | 64-chunk | 
# 4xGPU | 64-chunk | 

# 1xGPU | 128-chunk | 
# 2xGPU | 128-chunk | 
# 3xGPU | 128-chunk | 
# 4xGPU | 128-chunk | 

# 1xGPU | 256-chunk | 
# 2xGPU | 256-chunk | 
# 3xGPU | 256-chunk | 
# 4xGPU | 256-chunk | 


# zongyi.mp4 (141431 frames), person-n1 
# 1xGPU | 640-chunk | 45640 | ? | 414
# 2xGPU | 640-chunk | 
# 3xGPU | 640-chunk | 
# 4xGPU | 640-chunk | 136990 | 0.9834 | 174.15
