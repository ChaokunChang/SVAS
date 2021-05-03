import time
import sys
import logging
import numpy as np
import pandas as pd

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
        


class TopK:
    def __init__(self) -> None:
        self.selection_heap = None
        self.uncertain_table = None
        self.certain_table = None
        self.conf_thres = 0
        self.k = 0
        self.window = 0
        self.iter = 0
        pass

    def logic(self):
        while (self.dataframe.size > 0):
            # the frames to be cleaned in this iteration
            lucky_group = self.dataframe.select_topk(
                k=100, metric="proxy_score")
            oracle_labels = self.infer_engine.inference(
                "oracle", lucky_group)  # [[fid, label], ...]
            self.dataframe.update("oracle", oracle_labels)
            confidence_score = self.dataframe.get_confidence()
            if confidence_score >= self.conf_thres:
                break
        # evaluate and return
        eval = self.evaluate()
        return eval

    def run(self):
        prev_lam, prev_mu = -1, -1
        candidates = []
        while (self.selection_heap.size > 0):
            topk_prob = self.uncertain_table.topk_prob(self.certain_table.lam)
            if topk_prob >= np.log(self.conf_thres):
                break

            prev = time.time()
            if len(self.certain_table.topk) < self.k:
                print("bootstraping")
                clean_f = self.selection_heap.bootstrap(
                    self.k - len(self.certain_table.topk))
            else:
                candidates = self.selection_heap.select(topk_prob)
                clean_f = [sf.f for sf in candidates]

            if self.window < 2:
                scores = infer_frame_gt(f2idx(idx_list, clean_f), lr)
                for f, score in zip(clean_f, scores):
                    self.certain_table.insert_sf(
                        SF(score, f), self.uncertain_table.log_cdf[f])
                    mirrors = remained_ref[f, 2:]
                    for m in mirrors:
                        if m == -1:
                            break
                        self.certain_table.insert_mirror(SF(score, m))
            else:
                scores = infer_window_gt(
                    clean_f, self.window, window_samples, lr, data_size)
                for w, score in zip(clean_f, scores):
                    self.certain_table.insert_sw(
                        SF(score, w), self.uncertain_table.log_cdf[w])
            self.selection_heap.size -= len(scores)

            self.niter += 1

            if self.certain_table.lam != prev_lam or self.certain_table.mu != prev_mu:
                prev = time.time()
                self.selection_heap.update_order()
                prev_lam = self.certain_table.lam
                prev_mu = self.certain_table.mu

            if self.niter % 10 == 0:
                print('\rIter-{}, Ri: min {}, max {}, mean {}, Pi: {:.3f}, Ei: {:.3f}'
                      .format(self.niter, self.certain_table.topk[0].s, self.certain_table.topk[-1].s, self.certain_table.topk_mean(), topk_prob, candidates[0].s if len(candidates) != 0 else 0), end="")
                sys.stdout.flush()
        print('\nIter-{}, Ri: min {}, max {}, mean {}, Pi: {:.3f}, Ei: {:.3f}'
              .format(self.niter, self.certain_table.topk[0].s, self.certain_table.topk[-1].s, self.certain_table.topk_mean(), topk_prob, candidates[0].s if len(candidates) != 0 else 0))

        topk = list(reversed(self.certain_table.topk))

        precision, rank_dist, score_error = evaluate(
            topk, k, lr, self.window, data_size)
        print("[EXP]precision:", precision)
        print("[EXP]score_error:", score_error)
        print("[EXP]rank_dist:", rank_dist)
