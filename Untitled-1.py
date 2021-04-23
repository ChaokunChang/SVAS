#%%
print("Hello world")
# %%
import numpy as np
a = [[1,'a', 'A'], [2,'b', 'B'], [3, 'c', 'C']]
# print("a,",a[:,1])
b = np.array(a)
b[:,2].tolist()
# %%
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('info.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

logger.info('See the info.log file')
# %%
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info('Start reading database')

# read database here
records = {'krunal': 26, 'ankit': 24}
logger.debug('Records: %s', records)
logger.info('Updating records ...')
records = {'krunal': 27, 'ankit': 25}

# update records here
logger.info('Finish updating records')
# %%
[[]]*10
# %%
import ray
import asyncio
ray.init()

@ray.remote
class AsyncActor:
    # multiple invocation of this method can be running in
    # the event loop at the same time
    async def run_concurrent(self):
        print("started")
        await asyncio.sleep(2) # concurrent workload here
        print("finished")

actor = AsyncActor.remote()

# regular ray.get
ray.get([actor.run_concurrent.remote() for _ in range(4)])

# async ray.get
await actor.run_concurrent.remote()

# %%
import ray
from ray import serve
import requests
import random
import time

ray.init()
serve.start()

class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self, request):
        print(request.__dict__)
        user = request.query_params["name"]
        print("Input to Counter is: ", user)
        self.count += 1
        time.sleep(random.uniform(0, 3))
        return {"count": self.count, "user": user}


# Form a backend from our class and connect it to an endpoint.
serve.create_backend("my_backend", Counter)
serve.create_endpoint("my_endpoint", backend="my_backend", route="/counter")

# Query our endpoint in two different ways: from HTTP and from Python.

pool = []
iter_num = 5
iter = 0
count = 0
results = []
while (iter < iter_num):
    done, pool = ray.wait(pool)
    for i in range(len(done)):
        pool.append(serve.get_handle("my_endpoint").remote(name=f'foo{count}'))
        results.append(ray.get(done[i]))
        iter += 1
print(results)
# print(ray.get(serve.get_handle("my_endpoint").remote(name="foo2")))



# %%
