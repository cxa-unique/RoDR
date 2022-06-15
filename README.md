# RoDR

This repository contains the code and resources for our paper:

- Xuanang Chen, Jian Luo, Ben He, Le Sun, Yingfei Sun. 
Towards Robust Dense Retrieval via Local Ranking Alignment. In *IJCAI 2022*.

## Installation
Our code is developed based on [Tevatron](https://github.com/texttron/tevatron) toolkit.
We recommend you to create a new conda environment `conda create -n rodr python=3.7`, 
activate it `conda activate rodr`, and then install the following packages:
`torch==1.8.1`, `faiss-cpu==1.7.1`, `transformers==4.9.2`, `datasets==1.11.0`.

## Query Variations
> Note: In this repo, we mainly take MS MARCO passage ranking dataset for example, 
> and you can refer to `get_raw_data.sh` script to download the raw data saved in 
> the `msmarco_passage/raw` folder.

**Dev Query**: all query variation sets for `Dev` set used in our paper are provided 
in the `msmarco_passage/query/dev` folder. You can directly use them to test the DR model,
and you can also use `query_variation_generation.py` script to generate a query variation 
set by yourself:
```
qv_type=MisSpell
python query_variation_generation.py 
--original_query_file ./msmarco_passage/raw/queries.dev.small.tsv
--query_variation_file ./msmarco_passage/process/query/dev/queries.dev.small.${qv_type}.tsv
--variation_type ${qv_type}
```
You need to appoint `--variation_type` from `MisSpell, ExtraPunc, BackTrans, SwapSyn_Glove, 
SwapSyn_WNet, TransTense, NoStopword, SwapWords` eight types of query variations.
Note that a few queries can be kept original in the generated query variation set.
For example, if one query does not contain any stopword, the `NoStopword` variation is 
not applicable. 

**Train Query:** we also need to generate variations for train queries to enhance the DR model.
Similar to Dev set, we first generate eight variation sets for the train query set, and then merge
them together to obtain the final train query variation set, which is used to insert variations 
into the training data by adding a `'query_variation'` field into each training examples.
You can refer to `construct_train_query_variations.py` script after you obtain train variation sets and original training data.

## Training
**Standard DR:** To obtain a standard DR model, like `DR_OQ` in our paper, we need to
construct the training data first:
- `OQ`: the training data with original train queries using `bulid_train.py` script.
- `QV`: the training data with train query variations, by inserting the variation version 
of train queries into the `OQ` training data.

After that, you can refer to `train_standard_dpr.sh` script, to train the 
`DR_OQ`, `DR_QV`, and `DR_OQ->QV` models in our paper.

**RoDR:**
As for our proposed RoDR model, we first need to update the negatives in the `OQ` training data 
using the top candidates returned by `DR_OQ` model using `bulid_train_hn.py` script, 
and then insert the variation version of train queries into the updated training data.

After that, you can refer to `train_rodr_dpr.sh` script, to train the `RoDR w/ DR_OQ` model
on top of the `DR_OQ` model.

## Retrieval
After training a DR model, we can use it to carry out dense retrieval as follows:
1. Tokenizing: using `tokenize_passages.py` and `tokenize_queries.py` scripts to tokenize 
all passages in the corpus, and the dev queries and query variations.
2. Encoding and Retrieval: refer to `encode_retrieve_dpr.sh` to first encode passages and queries
into vectors, and then use Faiss to index and retrieve.

As for zero-shot retrieval on ANTIQUE, all DR models are only trained on MS MARCO passage dataset,
 please refer to `run_antique_zeroshot.sh` script.

## Resources
1. Query variations: 
    * Passage-Dev: available in the `msmarco_passage/query` folder, for both `dev` and `train` query sets.
    * Document-Dev: available in the `msmarco_doc/query` folder, for both `dev` and `train` query sets.
    * ANTIQUE: available in the `antique/query` folder, which are collected from [manually validated query variations](https://github.com/Guzpenha/query_variation_generators).

2. Models:

    | Dataset | DR_OQ | DR_QV | DR_OQ->QV | RoDR w/ OQ |
    |------------------|-----|-----|-----|-----|
    | MS MARCO Passage |  |  |  |  | 
    | MS MARCO Document |  |  |  |  |

3. Retrieval results:

    | Dataset | DR_OQ | DR_QV | DR_OQ->QV | RoDR w/ OQ | 
    |----------|-----|-----|-----|-----|
    | Passage-Dev |  |  |  |  | 
    | Document-Dev |  |  |  |  |
    | ANTIQUE |  |  |  |  | 

## Plugging into existing DR models
If you want to apply RoDR to publicly available DR models, such as ANCE, TAS-Balanced and ADORE+STAR, which are enhanced
in this paper, you need to make some minor changes in the model level, please refer to [here]() for more detailed instruction.
Herein, we only provide the model checkpoints and retrieval results.
1. Models:

    | ANCE | RoDR w/ ANCE | TAS-Balanced | RoDR w/ TAS-Balanced | ADORE+STAR | RoDR w/ ADORE+STAR | 
    |-----|-----|-----|-----|-----|-----|
    |  |  |  |  | 

2. Retrieval results:

    | Dataset | ANCE | RoDR w/ ANCE | TAS-Balanced | RoDR w/ TAS-Balanced | ADORE+STAR | RoDR w/ ADORE+STAR |
    |----------|-----|-----|-----|-----|-----|-----|
    | Passage-Dev |  |  |  |  | 
    | ANTIQUE |  |  |  |  | 
    
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