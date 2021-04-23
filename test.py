import ray
from ray import serve
import requests
import random
import time

ray.init()
serve.start()

class Counter:
    def __init__(self, model):
        self.count = 0
        self.model = model

    def __call__(self, request):
        # print(request.__dict__)
        user = request.query_params["name"]
        ctx = request.query_params["ctx"]
        print("Input to Counter is: ", user, " with context ", ctx, " with model ", self.model)
        self.count += 1
        time.sleep(random.uniform(0, 8))
        return {"model": self.model, "count": self.count, "user": user, "ctx": ctx}


# Form a backend from our class and connect it to an endpoint.
serve.create_backend("my_backend_0", Counter, "cmdn")
serve.create_backend("my_backend_1", Counter, "yolo")
serve.create_endpoint("my_endpoint_"+"cuda(0)", backend="my_backend_0", route="/counter1")
serve.create_endpoint("my_endpoint_"+"cuda(1)", backend="my_backend_1", route="/counter2")

class TaskManager:
    def __init__(self, ctx_list) -> None:
        self.deployments = [serve.get_handle("my_endpoint_1").remote, serve.get_handle("my_endpoint_2").remote, 
                            serve.get_handle("my_endpoint_3").remote, serve.get_handle("my_endpoint_4").remote]
        self.running_tasks = []
        self.ctx_list = []
        for ctx in ctx_list:
            self.deployments[ctx] = 0
        self.pool = [] # self.ctx_status
    
    def get_free_ctx(self):
        pass
    
    def get_free_server(self):
        return serve.get_handle("my_endpoint").remote
        pass

    def ctx2server(self, ctx):
        pass

    def server2ctx(self, server):
        pass

    def add(self, data):
        ctx = self.get_free_ctx()
        pool.append(self.get_free_server(ctx)(name='foo-2'))
        pass



deployment2ctx = {}
ctx2deployment = {}

pool = []
pool.append(serve.get_handle("my_endpoint_"+"cuda(0)").remote(name='foo-1', ctx = "cuda(0)"))
pool.append(serve.get_handle("my_endpoint_"+"cuda(1)").remote(name='foo-2', ctx = "cuda(1)"))
# pool.append(serve.get_handle("my_endpoint").remote(name='foo-3', ctx = "cuda(2)"))
# pool.append(serve.get_handle("my_endpoint").remote(name='foo-4', ctx = "cuda(3)"))
iter_num = 5
iter = len(pool)
results = []
while (iter <= iter_num):
    done, pool = ray.wait(pool)
    for i in range(len(done)):
        print(done[i].__dir__())
        r = ray.get(done[i])
        results.append(r)
        pool.append(serve.get_handle("my_endpoint_"+r['ctx']).remote(name=f'foo-{iter+1}', ctx = r['ctx']))
        iter += 1
print(results)