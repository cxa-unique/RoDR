import random
import argparse
import torch

def read_query(query_file):
    print('Loading original query ...')
    query_dict = {}
    with open(query_file, 'r') as f:
        for line in f:
            q_id, q_text = line.strip().split('\t')
            if q_id not in query_dict:
                query_dict[q_id] = q_text.strip()
            else:
                raise KeyError('Duplicated query ID, please check it.')
    print('Loading query original done!')
    return query_dict


def text_flint_modify(query_dict, variation_type):
    from textflint.input.component.sample.sm_sample import SMSample

    from textflint.generation.transformation.UT.keyboard import Keyboard
    from textflint.generation.transformation.UT.ocr import Ocr
    from textflint.generation.transformation.UT.typos import Typos
    from textflint.generation.transformation.UT.spelling_error import SpellingError
    from textflint.generation.transformation.UT.back_trans import BackTrans
    from textflint.generation.transformation.UT.swap_syn_word_embedding import SwapSynWordEmbedding
    from textflint.generation.transformation.UT.swap_syn_wordnet import SwapSynWordNet
    from textflint.generation.transformation.UT.tense import Tense

    ocr_trans = Ocr(trans_min=1, trans_max=1)
    keyboard_trans = Keyboard(trans_min=1, trans_max=1)
    spellerror_trans = SpellingError(trans_min=1, trans_max=1)
    typos_trans = Typos(trans_min=1, trans_max=1)
    if variation_type == 'MisSpell':
        trans = random.sample([ocr_trans, keyboard_trans, spellerror_trans, typos_trans], k=1)[0]
    elif variation_type == 'BackTrans':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trans = BackTrans(device=device)
    elif variation_type == 'SwapSyn_Glove':
        trans = SwapSynWordEmbedding(trans_min=1, trans_max=1)
    elif variation_type == 'SwapSyn_WNet':
        trans = SwapSynWordNet(trans_min=1, trans_max=1)
    elif variation_type == 'TransTense':
        trans = Tense(trans_min=1, trans_max=1)
    else:
        raise ValueError

    variation_dict = {}
    for q_id in query_dict:
        if q_id in variation_dict:
            raise ValueError
        q_text = query_dict[q_id]
        data = {'sentence1': q_text,
                'sentence2': "Empty",
                'y': '0'}
        sample = SMSample(data)
        trans_sample = trans.transform(sample, field='sentence1', n=1)
        if len(trans_sample) < 1:
            if variation_type == 'MisSpell':  # Try more times using other misspelling types
                no_trans = True
                for trans in [ocr_trans, keyboard_trans, spellerror_trans, typos_trans]:
                    trans_sample_1 = trans.transform(sample, field='sentence1', n=1)
                    if len(trans_sample_1) < 1:
                        continue
                    trans_text_1 = trans_sample_1[0].dump()['sentence1']
                    if trans_text_1 != q_text:
                        variation_text = trans_text_1
                        no_trans = False
                        break
                if no_trans:
                    print('Note: no modification to original query {} !'.format(q_id))
                    variation_text = q_text
            else:
                print('Note: no modification to original query {} !'.format(q_id))
                variation_text = q_text
        else:
            trans_text = trans_sample[0].dump()['sentence1']
            if trans_text == '' or trans_text == q_text:
                print('Note: no modification to original query {} !'.format(q_id))
                variation_text = q_text
            else:
                variation_text = trans_text

        variation_dict[q_id] = variation_text

    return variation_dict


def add_punc(query_dict):
    puncs = ['.', '..', '...',
             '!', '!!', '!!!',
             '?', '??', '???',
             ',', ',,', ',,,']

    variation_dict = {}
    for q_id in query_dict:
        if q_id in variation_dict:
            raise ValueError
        q_text = query_dict[q_id]
        add_punc = random.sample(puncs, k=1)[0]
        variation_text = q_text + add_punc
        variation_dict[q_id] = variation_text

    return variation_dict


def remove_stopwords(query_dict):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words('english'))

    variation_dict = {}
    for q_id in query_dict:
        if q_id in variation_dict:
            raise ValueError
        q_text = query_dict[q_id]
        word_tokens = word_tokenize(q_text)
        filtered_query = []

        for w in word_tokens:
            if w not in stop_words:
                filtered_query.append(w)

        variation_text = ' '.join(filtered_query)
        variation_dict[q_id] = variation_text
        if variation_text == q_text:
            print('Note: no modification to original query {} !'.format(q_id))

    return variation_dict


def reorder_words(query_dict):
    from textattack.transformations import WordInnerSwapRandom
    from textattack.augmentation import Augmenter

    augmenter = Augmenter(transformation=WordInnerSwapRandom(), transformations_per_example=1)
    variation_dict = {}
    for q_id in query_dict:
        if q_id in variation_dict:
            raise ValueError
        q_text = query_dict[q_id]
        variation_text = augmenter.augment(q_text)[0]
        try_num = 1
        while variation_text == q_text:  # Try more times
            variation_text = augmenter.augment(q_text)[0]
            try_num += 1
            if try_num > 10:
                print('Note: no modification to original query {} !'.format(q_id))
                break
        variation_dict[q_id] = variation_text

    return variation_dict


VARIATION_TYPES = ['MisSpell', 'ExtraPunc', 'BackTrans', 'SwapSyn_Glove', 'SwapSyn_WNet', 'TransTense', 'NoStopword', 'SwapWords']

def generate_query_variations(original_query_dict, variation_type, query_variation_file):
    assert variation_type in VARIATION_TYPES

    if variation_type in ['MisSpell', 'BackTrans', 'SwapSyn_Glove', 'SwapSyn_WNet', 'TransTense']:
        query_variation_dict = text_flint_modify(original_query_dict, variation_type)
    elif variation_type == 'ExtraPunc':
        query_variation_dict = add_punc(original_query_dict)
    elif variation_type == 'NoStopword':
        query_variation_dict = remove_stopwords(original_query_dict)
    elif variation_type == 'SwapWords':
        query_variation_dict = reorder_words(original_query_dict)
    else:
        raise NotImplementedError('Please select query variation type in [{}].'.format(VARIATION_TYPES))

    with open(query_variation_file, 'w') as w:
        for q_id in query_variation_dict:
            modified_q_text = query_variation_dict[q_id]
            w.write(q_id + '\t' + modified_q_text + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This script is used to generate query variation sets.')
    parser.add_argument('--original_query_file',
                        required=True,
                        help='The file that contains the original queries, e.g., queries.dev.small.tsv or train.query.tsv')
    parser.add_argument('--query_variation_file',
                        required=True,
                        help='The output file that contains the generated query variation sets: q_id \t q_text \n')
    parser.add_argument('--variation_type',
                        required=True,
                        help='The query variation type that you want to insert to the original query set.')

    args = parser.parse_args()
    original_query_dict = read_query(args.original_query_file)

    generate_query_variations(original_query_dict, args.variation_type, args.query_variation_file)