# RoDR

This repository contains the code and resources for our paper:

- Xuanang Chen, Jian Luo, Ben He, Le Sun, Yingfei Sun. 
Towards Robust Dense Retrieval via Local Ranking Alignment. In *IJCAI 2022*.

## Installation
Our code is developed based on [Tevatron](https://github.com/texttron/tevatron) DR training toolkit.
We recommend you to create a new conda environment `conda create -n rodr python=3.7`, 
activate it `conda activate rodr`, and then install the following packages:
`torch==1.8.1`, `faiss-cpu==1.7.1`, `transformers==4.9.2`, `datasets==1.11.0`.

## Query Variations
> Note: In this repo, we mainly take MS MARCO passage ranking dataset for example, 
> and you can refer to `download_raw_data.sh` script to download the raw data, which will be
> saved in the `data/msmarco_passage/raw` folder.

**Dev Query**: all query variation sets for MS MARCO small Dev set used in our paper are provided 
in the `data/msmarco_passage/query/dev` folder. You can directly use these query variation 
sets to test the robustness of your DR model, and you can also use `query_variation_generation.py` 
script to generate a query variation set by yourself:
```
qv_type=MisSpell
python query_variation_generation.py 
--original_query_file ./msmarco_passage/raw/queries.dev.small.tsv
--query_variation_file ./msmarco_passage/process/query/dev/queries.dev.small.${qv_type}.tsv
--variation_type ${qv_type}
```
You need to appoint `qv_type` from eight types of query variations: 
`MisSpell, ExtraPunc, BackTrans, SwapSyn_Glove, SwapSyn_WNet, TransTense, NoStopword, SwapWords`.
Note that a few queries can be kept original in a certain query variation set.
For example, if one query does not contain any stopword, the `NoStopword` variation is 
not applicable. Besides, before using the `query_variation_generation.py` script, you may need to install
[TextFlint](https://github.com/textflint/textflint), 
[TextAttack](https://github.com/QData/TextAttack), 
[NLTK](https://www.nltk.org/) toolkits.

**Train Query:** we also need to generate variations for train queries to enhance the DR model.
Similar to Dev set, we first generate eight variation sets for the train query set, and then merge
them uniformly to obtain the final train query variation set (our generated train query variation file
 is available in the `data/msmarco_passage/query/train` folder), which is used to insert variations 
into the training data, by adding a `'query_variation'` field into each training examples.
You can refer to `construct_train_query_variations.py` script after you obtain train variation sets 
and original training data.

## Training
**Standard DR:** To obtain a standard DR model, like `DR_OQ` in our paper, we need to
construct the training data first:
- `OQ`: the training data with original train queries, generated by `bulid_train.py` script.
- `QV`: the training data with train query variations, by inserting the variation version 
of train queries into the `OQ` training data.

After that, you can refer to `train_standard_dpr.sh` script, to train the 
`DR_OQ`, `DR_QV`, and `DR_OQ->QV` models as reported in our paper.

**RoDR:**
As for our proposed RoDR model, to achieve better alignment, we need to collect nearer neighbors 
for queries.
Specifically, we update the negatives in the `OQ` training data by sampling from the top 
candidates returned by `DR_OQ` model, and you can refer to `bulid_train_hn.py` script, 
wherein `--query_variation` requires the train query variation file.
Certainly, you can insert the variation version of train queries after constructing 
the training data, like `QV`, using `construct_training_data_with_variations` function in the 
`construct_train_query_variations.py` script.

After that, you can refer to `train_rodr_dpr.sh` script, to train a `RoDR w/ DR_OQ` model
on top of the `DR_OQ` model.

## Retrieval
After training a DR model, we can use it to carry out dense retrieval as follows:
1. Tokenizing: using `tokenize_passages.py` and `tokenize_queries.py` scripts to tokenize 
all passages in the corpus, the original queries and query variations.
2. Encoding and Retrieval: refer to `encode_retrieve_dpr.sh` to first encode passages and queries
into vectors, and then use Faiss to index and retrieve.

As for zero-shot retrieval on ANTIQUE, all DR models are only trained on MS MARCO passage dataset,
 please refer to `run_antique_zeroshot.sh` script.

## Resources
1. Query variations: 
    * Passage-Dev: available in the `data/msmarco_passage/query` folder, for both `dev` and `train` query sets.
    * Document-Dev: available in the `data/msmarco_doc/query` folder, for both `dev` and `train` query sets.
    * ANTIQUE: available in the `data/antique/query` folder, which are collected from 
    [manually validated query variations](https://github.com/Guzpenha/query_variation_generators).

2. Models:

    | MS MARCO Passage | MS MARCO Document |
    |------------|-----------|
    | [DR_OQ]()  | [DR_OQ]() |
    | [DR_QV]()  | [DR_QV]() |
    | [DR_OQ->QV]()  | [DR_OQ->QV]() |
    | [RoDR w/ DR_OQ]()  | [RoDR w/ DR_OQ]() |

3. Retrieval results:

    | Dataset | DR_OQ | DR_QV | DR_OQ->QV | RoDR w/ DR_OQ | 
    |----------|-----|-----|-----|-----|
    | Passage-Dev | [Download]() | [Download]() | [Download]() | [Download]() | 
    | Document-Dev | [Download]() | [Download]() | [Download]() | [Download]() |
    | ANTIQUE | [Download]() | [Download]() | [Download]() | [Download]() |

## RoDR on existing DR models
If you want to apply RoDR to publicly available DR models, such as ANCE, TAS-Balanced and ADORE+STAR, which are enhanced
in this paper, you need to make some minor changes in the model level, such as adding the pooler in ANCE, and using 
separate query and passage encoders.
Herein, we provide the model checkpoints and retrieval results for the reproducibility of our experiments and other research uses.
1. Models:

    | Original | RoDR |
    |-----|-----|
    | [ANCE]() | [RoDR w/ ANCE]() |
    | [TAS-Balanced]() | [RoDR w/ TAS-Balanced]() |
    | [ADORE+STAR]() | [RoDR w/ ADORE+STAR]() |

2. Retrieval results:

    | Model | Passage-Dev | ANTIQUE |
    |----------|-----|-----|
    | ANCE | [Download]() | [Download]() |
    | RoDR w/ ANCE | [Download]() | [Download]() |
    | TAS-Balanced | [Download]() | [Download]() |
    | RoDR w/ TAS-Balanced | [Download]() | [Download]() |
    | ADORE+STAR | [Download]() | [Download]() |
    | RoDR w/ ADORE+STAR | [Download]() | [Download]() |
    
## Citation
If you find our paper/resources useful, please cite:
```
@inproceedings{Chen2022_IJCAI,
 author = {Xuanang Chen and
           Jian Luo and
           Ben He and
           Le Sun and
           Yingfei Sun},
 title = {Towards Robust Dense Retrieval via Local Ranking Alignment},
 booktitle = {IJCAI 2022},
 year = {2022},
}
```