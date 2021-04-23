#%%
import numpy as np
a = np.array([1,2,3,4])
func = lambda x: x>2
b = func(a)
print(b)
# %%
c = np.array(['a', 'b', 'c', 'd'])
c[b]
# %%
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

@ray.remote
class NoScopeService:
    def __init__(self, sid, video_file, ctx, img_size=(416, 416), num_threads=2):
        logging.basicConfig(level=logging.DEBUG, filename=f'./info{sid}.log')
        self.logger = logging.getLogger(__name__)
        self.sid = sid

    def __call__(self, request):
        offset = request.query_params["offset"]
        length = request.query_params["length"]
        self.logger.info(f'__call__ is called with {offset}, {length}.')
        targets, _ = self.run(offset=offset, length=length)
        return {"sid": self.sid, "results": targets}
    
    def get_mask(self, values, masker):
        return masker(values)

    def run(self, offset, length):
        self.logger.info(f'running on {self.sid} with params {offset}, {length}.')
        num_selected = random.randrange(0, length)
        candidates = list(range(offset, min(offset+length, 108000)))
        random.shuffle(candidates)
        targets = candidates[:num_selected]
        time.sleep(0.1 * num_selected)
        self.logger.info(f'Got {num_selected} items from {self.sid}.')
        info = {'sid': self.sid}
        return targets, info

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

servers = [NoScopeService.remote(sid=i, video_file="./input.mp4", ctx=f'cuda({i})') for i in range(4)]

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
    done, task_queue = ray.wait(task_queue)
    
    selected_ids, info = ray.get(done)[0]
    results += selected_ids
    free_server = servers[info['sid']]

    logger.info(f'***** iter-{niter}/{num_iters} with sid={info["sid"]} *****')
    offset = niter * chunk_size
    task_queue.append(free_server.run.remote(offset, chunk_size))
    niter += 1

logger.info(f'Job Finished, {len(results)} frames were selected. {results}')
