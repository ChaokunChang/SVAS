class BaseModel:
    def __init__(self) -> None:
        pass

    def infer(self, input):
        return {"label": "test", "prob": 1.0}
    
    def infer_batch(self, batch):
        ret = []
        for input in  batch:
            ret.append({"label": "test", "prob": 1.0})

    def build_proxy(self, data):
        # build a proxy model based on current model, using the provided data
        pass

    def benchmark(self, data):
        # Performance test of model
        pass

    def __call__(self, batch):
        return self.infer_batch(batch)
