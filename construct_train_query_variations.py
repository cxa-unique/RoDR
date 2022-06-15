from transformers import AutoTokenizer
import json
import random
import os
import argparse

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
variation_types = ['MisSpell', 'ExtraPunc', 'BackTrans', 'SwapSyn_Glove', 'SwapSyn_WNet', 'TransTense', 'NoStopword', 'SwapWords']
pass_variation_num = {'TransTense': 50097,
                      'NoStopword': 50098,
                      'SwapSyn_Glove': 50098,
                      'BackTrans': 50098,
                      'SwapSyn_WNet': 50098,
                      'SwapWords': 50098,
                      'MisSpell': 50098,
                      'ExtraPunc': 50097}

## Used for MS MARCO document dataset
# doc_variation_num = {'TransTense': 45773,
#                      'BackTrans': 45773,
#                      'SwapSyn_WNet': 45773,
#                      'NoStopword': 45773,
#                      'SwapSyn_Glove': 45774,
#                      'MisSpell': 45774,
#                      'SwapWords': 45773,
#                      'ExtraPunc': 45773}

def check_query_tokens(original_query_file, query_variation_dir):
    original_query = {}
    with open(original_query_file) as f:
        for line in f:
            q_id, q_text = line.strip('\n').split('\t')
            original_query[q_id] = q_text

    output_dir = os.path.join(query_variation_dir, 'changes_in_token')
    os.makedirs(output_dir, exist_ok=True)
    for qv_type in variation_types:
        query_variation_file = os.path.join(query_variation_dir, 'train.query.{}.tsv'.format(qv_type))
        output_variation_file = os.path.join(output_dir, 'train.query.{}.tsv'.format(qv_type))
        with open(query_variation_file) as f, \
            open(output_variation_file, 'w') as w:
            for line in f:
                q_id, q_text = line.strip('\n').split('\t')
                n_q_tokens = tokenizer.encode(q_text, add_special_tokens=False, max_length=128, truncation=True)
                o_q_tokens = tokenizer.encode(original_query[q_id], add_special_tokens=False, max_length=128, truncation=True)
                if n_q_tokens != o_q_tokens:
                    w.write(line)


def sample_train_query_variations(original_query_file, query_variation_dir):
    variation_select_dict = {}
    variation_dict = {}
    selected = []
    selected_dict = {}
    output_dir = os.path.join(query_variation_dir, 'changes_in_token')

    for qv_type in variation_types:
        query_variation_file = os.path.join(output_dir, 'train.query.{}.tsv'.format(qv_type))
        variation_dict[qv_type] = {}
        variation_select_dict[qv_type] = []
        with open(query_variation_file) as f:
            for line in f:
                q_id, q_text = line.strip().split('\t')
                if q_id in variation_dict[qv_type]:
                    raise ValueError
                variation_dict[qv_type][q_id] = q_text

        wait_sampled_list = list(variation_dict[qv_type].keys())
        for q in selected:
            if q in wait_sampled_list:
                wait_sampled_list.remove(q)
        if len(wait_sampled_list) < pass_variation_num[qv_type]:
            raise ValueError

        sample_q_list = random.sample(wait_sampled_list, k=pass_variation_num[qv_type])
        for sample_q in sample_q_list:
            assert sample_q not in selected
            variation_select_dict[qv_type].append(sample_q)
            selected.append(sample_q)
            if sample_q in selected_dict:
                raise ValueError
            selected_dict[sample_q] = variation_dict[qv_type][sample_q]
        assert len(variation_select_dict[qv_type]) == pass_variation_num[qv_type]

    with open(original_query_file) as f, \
        open(os.path.join(output_dir, 'train.query.variations.tsv'), 'w') as w:
        for line in f:
            q_id, q_text = line.strip().split('\t')
            assert q_text.lower() != selected_dict[q_id].lower()
            w.write(q_id + '\t' + selected_dict[q_id] + '\n')


def construct_training_data_with_variations(original_query_file, query_variation_dir, negatives_file, train_data_dir, output_train_data_dir):
    original_query = {}
    with open(original_query_file) as f:
        for line in f:
            q_id, q_text = line.strip('\n').split('\t')
            original_query[q_id] = q_text

    query_order_list = []  ## the query order in the training data
    with open(negatives_file) as f:
        for line in f:
            q, _ = line.strip().split('\t')
            query_order_list.append(q)

    query_variation_file = os.path.join(query_variation_dir, 'changes_in_token', 'train.query.variations.tsv')
    query_variation = {}
    with open(query_variation_file) as f:
        for line in f:
            q_id, q_text = line.strip('\n').split('\t')
            query_variation[q_id] = q_text

    os.makedirs(output_train_data_dir, exist_ok=True)
    q_num = 0
    for i in range(9):
        with open(train_data_dir + '/split0{}.json'.format(i)) as f, \
            open(output_train_data_dir + '/split0{}.json'.format(i), 'w') as w:

            for line in f:
                train_example = json.loads(line)
                qid = query_order_list[q_num]
                o_q_text = original_query[qid]
                o_q_tokens = tokenizer.encode(o_q_text, add_special_tokens=False, max_length=128, truncation=True)
                assert o_q_tokens == train_example['query']

                n_q_text = query_variation[qid]
                n_q_tokens = tokenizer.encode(n_q_text, add_special_tokens=False, max_length=128, truncation=True)
                assert n_q_tokens != o_q_tokens
                train_example['query_variation'] = n_q_tokens
                q_num += 1

                w.write(json.dumps(train_example) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is used to construct training data with query variations.')
    parser.add_argument('--original_query_file',
                        required=True,
                        help='The file that contains the original queries, e.g., train.query.tsv')
    parser.add_argument('--query_variation_dir',
                        required=True,
                        help='The path that contains the generated various query variation sets: q_id \t q_text \n')
    parser.add_argument('--negatives_file',
                        required=True,
                        help='The negatives file that corresponds to the training data.')
    parser.add_argument('--train_data_dir',
                        required=True,
                        help='The training data dir that contains separate json file.')
    parser.add_argument('--output_train_data_dir',
                        required=True,
                        help='The output training data dir that has been inserted with query variations.')

    args = parser.parse_args()

    ## Step 1: find all train queries that have been modified in token level.
    check_query_tokens(args.original_query_file, args.query_variation_dir)

    ## Step 2: sample and merge various train query variations.
    sample_train_query_variations(args.original_query_file, args.query_variation_dir)

    ## Step 3: insert query variations into the training data.
    construct_training_data_with_variations(args.original_query_file, args.query_variation_file, args.negatives_file,
                                            args.train_data_dir, args.output_train_data_dir)