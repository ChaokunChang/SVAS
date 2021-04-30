import numpy as np

class CachedGTLabelReader():
    def __init__(self, cached_gt_path, offset=0):
        self.cached_gt = np.load(cached_gt_path)
        self.offset = offset

    def __len__(self):
        return len(self.cached_gt)

    def __getitem__(self, idx):
        return self.cached_gt[idx + self.offset]

    def get_batch(self, batch):
        batch = [idx + self.offset for idx in batch]
        return self.cached_gt[batch]

class ForceVideoLabelReader():
    def __init__(self, model, video_loader, offset=0):
        self.model = model
        self.raw_data = video_loader
        self.offset = offset
        self.cache = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            frame = self.raw_data[idx + self.offset]
            predict = self.model(frame)
            self.cache[idx] = predict
            return predict

    def get_batch(self, batch):
        batch = [idx + self.offset for idx in batch]
        cached = []
        non_cached = []
        results = []
        for i, idx in enumerate(batch):
            if idx in self.cache:
                cached.append(i)
                results.append(self.cache[idx])
            else:
                non_cached.append(i)
                results.append(None)
        frames = self.raw_data.get_batch(non_cached)
        predicts = self.model(frames)
        for i, predict in enumerate(predicts):
            results[non_cached[i]] = predict
        return results