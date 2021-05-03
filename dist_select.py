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

from dataloader.video_loader import DecordVideoReader
from models.detector import YOLOv5
from models.difference import SimpleDiff
from models.proxy import tinyresnet18, ProxyTinyResNet
from utils.data_processing import get_tmp_results_path

ray.init()
serve.start()


class Model:
    def __init__(self) -> None:
        pass

    def infer_batch(self):
        pass

    def __call__(self):
        # split batch automatically
        # and inference batch by batch to fully utilize the hardware
        self.infer_batch()

    def train(self, model_path=None):
        pass


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
        self.oracle_score_thresh = opt.oracle_score_thresh
        self.oracle = YOLOv5(model_type="yolov5x", thr=self.oracle_score_thresh,
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
        print("DEBUG3: ", proxy_scores)
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
            b_labels = self.oracle.infer4object(batch)
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
    parser.add_argument("--diff_delay", type=int,
                        help="distance to compute difference of two frames, also sampling frame rate", default=30)

    # Proxy Model
    parser.add_argument("--proxy_model_ckpt", type=str, default="data/videos/example/checkpoint_final.pt",
                        help="checkpoiny of pre-trained proxy model")
    parser.add_argument("--proxy_batch_size", type=int, default=64)
    parser.add_argument("--proxy_score_upper", type=float, default=0.9)
    parser.add_argument("--proxy_score_lower", type=float, default=0.1)

    # Oracle Model
    # A better design is oracle model is yolov5-2, which means a special model for yolov5 which output car only
    parser.add_argument("--oracle_model", type=str, default="yolov5",
                        choices=['yolov5'], help="model of oracle")
    parser.add_argument("--oracle_batch_size", type=int, default=16)
    parser.add_argument("--oracle_target_label", type=int,
                        default=2, help="2 means car")
    parser.add_argument("--oracle_score_thresh", type=float, default=0.25)

    # Scheduler
    parser.add_argument("--chunk_size", type=int, default=128)
    parser.add_argument("--num_gpus", type=int, default=1)

    # Worker
    # pass

    # Other
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--save", default=False,
                        help="save intermediate results", action="store_true")

    opt, _ = parser.parse_known_args()
    opt.video = os.path.abspath(opt.video)

    return opt


if __name__ == "__main__":
    opt = get_options()
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    num_servers = opt.num_gpus
    servers = []
    for i in range(num_servers):
        servers.append(NoScopeService.remote(sid=i, opt=opt))

    video_length = opt.length
    chunk_size = opt.chunk_size
    num_iters = video_length // chunk_size
    niter = 0
    task_queue = []
    results = []
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
        logger.debug(f'Iter-{niter}/{num_iters} Got {len(selected_ids)} items from server-{info["sid"]}, {info}')
        free_server = servers[info['sid']]
        task_queue.append(free_server.run.remote(niter*chunk_size, chunk_size))
        niter += 1

    while len(task_queue) > 0:
        done, task_queue = ray.wait(task_queue)
        selected_ids, info = ray.get(done)[0]
        results += selected_ids

    end_time = time.time()
    logger.info(
        f'Job Finished, {len(results)} frames were selected from {video_length} with cost of {end_time - start_time}s.')
    
    np.save(get_tmp_results_path(opt.video), np.array(results))

# Exp
# 1xGPU 64-chunk 15296 155
# 2xGPU 64-chunk 15296 155
# 3xGPU 64-chunk 15296 142
# 4xGPU 64-chunk 15296 142

# 1xGPU 128-chunk 15296 
# 2xGPU 128-chunk 15296 
# 3xGPU 128-chunk 15296 
# 4xGPU 128-chunk 15296 

# 1xGPU 256-chunk 15296 114
# 2xGPU 256-chunk 15296 
# 3xGPU 256-chunk 15296 
# 4xGPU 256-chunk 15296 91

# 1xGPU 640-chunk 15296 155
# 2xGPU 640-chunk 15296 155
# 3xGPU 640-chunk 15296 142
# 4xGPU 640-chunk 15296 82
