
from numpy.core.numeric import Inf
import torch

class SimpleDiff():
    def __init__(self, diff_thresh=30.0, delay=30) -> None:
        self.diff_thresh = diff_thresh
        self.delay = delay
    
    def infer(self, frames, delay=None):
        if delay is None:
            delay = self.delay
        num_frames = len(frames)
        results = [Inf] + [-Inf]*(min(delay, num_frames)-1)
        prev_key_id = len(results) // 2
        ref_frame = frames[prev_key_id]
        for start_id in range(delay, num_frames, delay):
            end_id = min(start_id + delay, num_frames)
            batch_results = [-Inf for _ in range(start_id, end_id)]
            if len(batch_results) == 0:
                break
            cur_key_id = (start_id + end_id) // 2
            cur_frame = frames[cur_key_id]
            diff = ((cur_frame - ref_frame.unsqueeze(0))**2).view(len(cur_frame), -1).mean(-1)
            score = torch.sum(diff).cpu()
            # print("DEBUG-DIFF: ", diff)
            if score > self.diff_thresh:
                batch_results[0] = Inf
                ref_frame = cur_frame
            results.extend(batch_results)
        return results
    
    def __call__(self, frames, delay=None):
        return self.infer(frames, delay)
