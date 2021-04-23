from numpy.core import numeric
from numpy.core.numeric import Inf
from numpy.lib.function_base import diff
import ray
from ray import serve
import requests
import random
import time
import decord
import logging

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


@ray.remote
class NoScopeService:
    def __init__(self, sid, video_file, ctx, img_size=(416, 416), num_threads=2):
        self.sid = sid
        self.count = 0
        decord_ctx = decord.cpu()
        if ctx == "gpu(0)":
            decord_ctx = decord.gpu(0)
        elif ctx == "gpu(1)":
            decord_ctx = decord.gpu(1)
        elif ctx == "gpu(2)":
            decord_ctx = decord.gpu(2)
        elif ctx == "gpu(3)":
            decord_ctx = decord.gpu(3)
        else:
            raise NotImplementedError
        self.videoloader = decord.VideoReader(video_file, ctx=decord_ctx, width=img_size[0], height=img_size[1], num_threads=num_threads)
        self.diff_detector = None # load difference detector model
        self.proxy = None # load proxy model
        self.oracle = None # load oracle model
        self.batch_size = 8

        logging.basicConfig(level=logging.DEBUG, filename=f'./info{sid}.log')
        self.logger = logging.getLogger(__name__)

    def __call__(self, request):
        offset = request.query_params["offset"]
        length = request.query_params["length"]
        targets, _ = self.run(offset=offset, length=length)
        return {"sid": self.sid, "results": targets}
    
    def get_mask(self, values, masker):
        return masker(values)

    def run(self, offset, length):
        targets = []
        c_low = 0.10
        c_high = 0.99
        diff_thres = 30.0
        batch_fids = list(range(offset, min(offset+length, len(self.videoloader))))
        frames = self.videoloader.get_batch(batch_fids) # return ndarray

        candidates = list(range(len(batch_fids)))
        
        distances = self.diff_detector(frames, delay=1)
        diff_mask = self.get_mask(distances, lambda x: x >= diff_thres)
        frames = frames[diff_mask] # mask on frames
        candidates[diff_mask]
        
        proxy_scores = self.proxy(frames)
        proxy_mask = self.get_mask(proxy_scores, lambda x: x > c_low and x < c_high)
        candidate_from_proxy = (candidates[self.get_mask(proxy_scores, lambda x: x >= c_high)])
        frames = frames[proxy_mask]
        candidates = candidates[proxy_mask]

        labels = self.oracle(frames)
        oracle_mask = self.get_mask(labels, lambda x: x >= 0.5)
        frames = frames[oracle_mask]
        candidates = candidates[proxy_mask]

        candidates = candidates + candidate_from_proxy
        for i in candidates:
            cur = i
            targets.append(batch_fids[cur])
            cur += 1
            while diff_mask[cur] == 0:
                targets.append(batch_fids[cur])
        
        info = {'sid': self.sid}
        return targets, info 

    def infer_batch(self, offset, length):
        batch_num = length / self.batch_size
        targets = []
        c_low = 0.10
        c_high = 0.99
        diff_thres = 30.0
        for i in range(batch_num):
            batch_fids = range(offset + i*self.batch_size, offset + (i+1)*self.batch_size)
            frames = self.videoloader.get_batch(batch_fids) # return ndarray
            candidates = range(len(batch_fids))
            
            distances = self.diff_detector(frames, delay=1)
            diff_mask = self.get_mask(distances, lambda x: x >= diff_thres)
            frames = frames[diff_mask] # mask on frames
            candidates[diff_mask]
            
            proxy_scores = self.proxy(frames)
            proxy_mask = self.get_mask(proxy_scores, lambda x: x > c_low and x < c_high)
            candidate_from_proxy = (candidates[self.get_mask(proxy_scores, lambda x: x >= c_high)])
            frames = frames[proxy_mask]
            candidates = candidates[proxy_mask]

            labels = self.oracle(frames)
            oracle_mask = self.get_mask(labels, lambda x: x >= 0.5)
            frames = frames[oracle_mask]
            candidates = candidates[proxy_mask]

            candidates = candidates + candidate_from_proxy
            for i in candidates:
                cur = i
                targets.append(batch_fids[cur])
                cur += 1
                while diff_mask[cur] == 0:
                    targets.append(batch_fids[cur])

        return targets

    def infer_once(self, offset, length):
        targets = []
        c_low = 0.10
        c_high = 0.99
        diff_thres = 30.0
        last_frame = None
        last_label = None
        for i in range(length):
            fid = offset + i
            cur_frame = self.videoloader[fid]
            if last_frame is None:
                last_frame = -cur_frame*Inf
                last_label = 0
            
            if (self.diff_detector([last_frame, cur_frame])[1] < diff_thres):
                cur_label = last_label
            else:
                proxy_score = self.proxy(cur_frame)
                if proxy_score <= c_low:
                    cur_label = 0
                elif proxy_score >= c_high:
                    cur_label = 1
                else:
                    cur_label = self.oracle(cur_frame) >= 0.5
            
            if cur_label == 1:
                targets.append(fid)
            
            last_frame = cur_frame
            last_label = cur_label

        return targets


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

servers = [NoScopeService.remote(sid=i, video_file="/research/dept8/csci5570/ckchang/everest/videos/traffic_footage_0-1h.mp4", ctx=f'gpu({i})') for i in range(4)]

video_length = 108000
chunk_size = 128
num_iters = video_length // chunk_size
niter = 0
task_queue = []
results = []
while niter < len(servers):
    logger.info(f'***** iter-{niter}/{num_iters} with sid={niter} *****')
    offset = niter * chunk_size
    task_queue.append(servers[niter].run.remote(offset, chunk_size))
    niter += 1

while niter < num_iters:
    # wait will block until the first task finished.
    done, running = ray.wait(task_queue)
    
    selected_ids, info = ray.get(done)[0]
    results += selected_ids
    logger.info(f'Got {len(selected_ids)} items from server-{info["sid"]}. ')
    free_server = servers[info['sid']]
    logger.info(f'***** iter-{niter}/{num_iters} with sid={info["sid"]} *****')
    task_queue.append(free_server.run.remote(niter*chunk_size, chunk_size))
    niter += 1

logger.info(f'Job Finished, {len(results)} frames were selected. {results}')