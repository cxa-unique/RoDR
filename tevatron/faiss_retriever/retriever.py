import numpy as np
import faiss
import re

import logging
logger = logging.getLogger(__name__)


def index_from_factory(init_reps: np.ndarray, index_param: str):
    # index_param: Flat; PQ48; SQ8; PCAR64,Flat
    # see more in https://github.com/facebookresearch/faiss/wiki/The-index-factory
    if '-' in index_param:  # PCAR64-Flat
        index_param = re.sub('-', ',', index_param)
    dim, measure = init_reps.shape[1], faiss.METRIC_INNER_PRODUCT
    index = faiss.index_factory(dim, index_param, measure)

    if not index.is_trained:
        index.train(init_reps)
    logger.info(index.is_trained)
    index.add(init_reps)
    return index


class BaseFaissIPRetriever:
    def __init__(self, init_reps: np.ndarray, index_type: str, index_file: str, construct: bool):
        if construct:
            self.index = index_from_factory(init_reps, index_type)
        else:
            self.index = faiss.read_index(index_file)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def save_index(self, save_dir: str):
        faiss.write_index(self.index, save_dir)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in range(0, num_query, batch_size):
            logger.info("Searching idx: {}".format(start_idx))
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices