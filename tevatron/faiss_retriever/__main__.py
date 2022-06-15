import torch
import numpy as np
from argparse import ArgumentParser
import os
from .retriever import BaseFaissIPRetriever

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            for r, (s, idx) in enumerate(score_list):   ## saving in TREC format
                f.write('\t'.join([str(qid), 'Q0', str(idx), str(r+1), str(s), 'dense']) + '\n')

def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[p_lookup[x] for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices

def main():
    parser = ArgumentParser()
    parser.add_argument('--query_reps', required=True)
    parser.add_argument('--passage_reps', required=False, default=None)
    parser.add_argument('--index_type', required=False, default=None)
    parser.add_argument('--passage_index', required=False, default=None)
    parser.add_argument('--passage_lookup', required=False, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--depth', type=int, default=1000)
    parser.add_argument('--save_ranking_file', required=True)
    parser.add_argument('--save_index', action='store_true')
    parser.add_argument('--save_index_dir', required=False, default=None)

    args = parser.parse_args()
    logger.info("Parameters %s", args)

    if args.passage_reps is None and args.passage_index is None:
        raise RuntimeError('Reps and Index must be given one!')

    save_ranking_path = '/'.join(args.save_ranking_file.split('/')[:-1])
    os.makedirs(save_ranking_path, exist_ok=True)

    if args.passage_index:  ## read index from pre-constructed file
        assert args.passage_lookup is not None
        retriever = BaseFaissIPRetriever(init_reps=np.array(0), index_type='', construct=False, index_file=args.passage_index)
        p_lookup = torch.load(args.passage_lookup)
    else:  ## construct index from reps file
        assert args.passage_reps is not None
        p_reps, p_lookup = torch.load(args.passage_reps)
        retriever = BaseFaissIPRetriever(init_reps=p_reps.float().numpy(), index_type=args.index_type, construct=True, index_file='')
        if args.save_index:
            assert args.save_index_dir is not None
            os.makedirs(args.save_index_dir, exist_ok=True)
            retriever.save_index(os.path.join(args.save_index_dir, args.index_type + '.index'))
            torch.save(p_lookup, os.path.join(args.save_index_dir, 'p_lookup.pt'))

    q_reps, q_lookup = torch.load(args.query_reps)
    q_reps = q_reps.float().numpy()

    logger.info('Index Search Start')
    all_scores, psg_indices = search_queries(retriever, q_reps, p_lookup, args)
    logger.info('Index Search Finished')

    write_ranking(psg_indices, all_scores, q_lookup, args.save_ranking_file)


if __name__ == '__main__':
    main()