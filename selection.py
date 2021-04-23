import time
import sys
import logging
import numpy as np
import pandas as pd
import ray
import asyncio

logging.basicConfig(level=logging.DEBUG, filename="./running.log")
logger = logging.getLogger(__name__)

class VideoVirtualRelation:
    def __init__(self, pdframe) -> None:
        self._confidence_score
        self._updated = True
        self.dataset = pdframe  # id, group, label, proxy_score, rank, etc
        self.sampled_dataset = None  # from self.dataset
        self.uncertain_dataset = None  # from self.dataset
        self.certain_dataset = None  # from self.dataset
        pass

    def select_topk(self, k=1, metric=None):
        pass

    def update(self, column, keys, values):
        self._updated = True
        pass

    def get_confidence(self):
        self._updated = False
        pass

    def get_all_ids(self):
        return self.dataset['id']

    def size(self):
        return len(self.dataset)

    def __len__(self):
        return self.size()

    @property
    def confidence_score(self):
        if self._updated:
            self._confidence_score = self.get_confidence()
        return self._confidence_score


class VideoInferenceEngine:
    def __init__(self, data) -> None:
        pass

    def infernce(self, models, data_ids, ctx=None):
        # ctx=None means automatically choosing
        pass

    def dist_infer(self, config):
        # config: [{}]
        pass


class Selection:
    def __init__(self, dataframe: VideoVirtualRelation, infer_engine) -> None:
        self.dataframe = dataframe
        self.infer_engine = infer_engine

    def scan(self):
        # Scan the video to tag each frame with:
        # 1. A group number with is determined by difference detector
        # 2. The prediction score of proxy model (light weighr model)
        frame_ids = self.dataframe.get_all_ids()
        # return [[fid, group_id, proxy_score], ...]
        infer_results = self.infer_engine.inference(
            models=["diff_detector", "proxy"], data_ids=frame_ids)

        group_ids = np.array(infer_results)[:, 1].tolist()
        proxy_scores = np.array(infer_results)[:, 2].tolist()
        self.dataframe.update(column="proxy_score",
                              keys=frame_ids, values=group_ids)
        self.dataframe.update(
            column="group_id", keys=frame_ids, values=proxy_scores)
        logging.info(
            f'{len(infer_results)} / {len(self.dataframe)} was scaned with difference detector and proxy model.')
        return len(infer_results)

    def run(self):
        processed_num = 0
        iter_num = len(self.dataframe) // self.block_size
        iter = 0
        results = [[]] * iter_num
        while(iter < iter_num):
            data_batch = self.dataframe.get_batch(
                offset=processed_num, batch_size=128)
            ctx = self.infer_engine.get_free_ctx()
            candidates = self.infer_engine.inference(
                models=["NoScope-Pipeline"], data_ids=data_batch, ctx=ctx)
            results[iter] = candidates
            iter += 1
            processed_num += len(data_batch)
        