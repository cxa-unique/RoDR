# RoDR

This repository contains the code and resources for our paper:

- Xuanang Chen, Jian Luo, Ben He, Le Sun, Yingfei Sun. 
[Towards Robust Dense Retrieval via Local Ranking Alignment](https://www.ijcai.org/proceedings/2022/0275.pdf). In *IJCAI 2022*.

![image](https://github.com/cxa-unique/RoDR/blob/main/rodr_framework.png)

## Installation
Our code is developed based on [Tevatron](https://github.com/texttron/tevatron) DR training toolkit.
We recommend you to create a new conda environment `conda create -n rodr python=3.7`, 
activate it `conda activate rodr`, and then install the following packages:
`torch==1.8.1`, `faiss-cpu==1.7.1`, `transformers==4.9.2`, `datasets==1.11.0`.

## Query Variations
> Note: In this repo, we mainly take MS MARCO passage ranking dataset for example. Before the experiments,  
> you can refer to `download_raw_data.sh` script to download and process the raw data, which will be
> saved in the `data/msmarco_passage/raw` folder, like `train.negatives.tsv` file that
> contains the negatives of each train query for constructing the training data.

**Dev Query**: All query variation sets for MS MARCO small Dev set used in our paper are provided 
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
You need to appoint the type of query variation (namely, `qv_type`) from pre-defined eight types of query variations: 
`MisSpell, ExtraPunc, BackTrans, SwapSyn_Glove, SwapSyn_WNet, TransTense, NoStopword, SwapWords`.
Note that a few queries can be kept original in a certain query variation set.
For example, if one query does not contain any stopword, the `NoStopword` variation is 
not applicable. Besides, before using the `query_variation_generation.py` script, you may need to install
[TextFlint](https://github.com/textflint/textflint), 
[TextAttack](https://github.com/QData/TextAttack), 
[NLTK](https://www.nltk.org/) toolkits.

**Train Query:** We also need to generate variations for train queries to enhance the DR model.
Similar to Dev set, we first generate eight variation sets for the train query set, and then merge
them uniformly to obtain the final train query variation set (our generated train query variation file
 is available in the `data/msmarco_passage/query/train` folder), which is used to insert variations 
into the training data, by adding a `'query_variation'` field into each training examples.
You can refer to `construct_train_query_variations.py` script after you obtain train variation sets 
and original training data.

## Training
**Standard DR:** To obtain a standard DR model, like `DR_OQ` in our paper, you need to
construct the training data first:
- `OQ`: the training data with original train queries, generated by `bulid_train.py` script.
- `QV`: the training data with train query variations, by inserting the variation version 
of original train queries into the `OQ` training data.

After that, you can refer to `train_standard_dpr.sh` script, to train the 
`DR_OQ`, `DR_QV`, and `DR_OQ->QV` models using the `OQ` and `QV` training data 
as described in our paper.

**RoDR:**
As for our proposed RoDR model, to achieve better alignment, you need to collect nearer neighbors 
for queries. Specifically, you can update the negatives in the `OQ` training data by sampling from 
the top candidates returned by `DR_OQ` model. After that, you can refer to `bulid_train_nn.py` 
script, wherein `--query_variation` argument requires the generated train query variation file.
Certainly, you can also add the variation version of train queries after constructing 
the training data, similar to `QV`, using `construct_training_data_with_variations` function 
available in the `construct_train_query_variations.py` script.

After that, you can refer to `train_rodr_dpr.sh` script, to train a `RoDR w/ DR_OQ` model
on top of the `DR_OQ` model. Compared to standard DR training, you need to change `--training_mode`
to `oq.qv.lra` mode, provide the initial DR model path to `--model_name_or_path` argument, and set
the loss weights in Eq. 8, as described in our paper.

## Retrieval
After training a DR model, you can use it to carry out dense retrieval as follows:
1. Tokenizing: using `tokenize_passages.py` and `tokenize_queries.py` scripts to tokenize 
all passages in the corpus, the original queries and query variations.
2. Encoding and Retrieval: refer to `encode_retrieve_dpr.sh` to first encode passages and queries
into vectors, and then use [Faiss](https://github.com/facebookresearch/faiss) to index and retrieve.

As for zero-shot retrieval on ANTIQUE, all DR models are only trained on MS MARCO passage dataset,
 please refer to `run_antique_zeroshot.sh` script.

For the evaluation on MS MARCO passage ranking dataset, such as MRR@10, Recall, and statistical t-test, 
we provide `variations_avg_tt_test.py` script to compute the metrics for all paired run files 
from two DR models waiting for comparison. You can use it like this: 
```
# for single run file
python variations_avg_tt_test.py qrels run_file1 run_file2
# for all run files
python variations_avg_tt_test.py qrels run_dir1 run_dir2 fusion
```

## Resources
1. Query variations: 
    * Passage-Dev: available in the `data/msmarco_passage/query` folder, for both `dev` and `train` query sets.
    * Document-Dev: available in the `data/msmarco_doc/query` folder, for both `dev` and `train` query sets.
    * ANTIQUE: available in the `data/antique/query` folder, which are collected from five types of 
    [manually validated query variations](https://github.com/Guzpenha/query_variation_generators).

2. Models:

    | MS MARCO Passage | MS MARCO Document |
    |------------|-----------|
    | [DR_OQ](https://drive.google.com/file/d/1CEV-nCY3r2-HXusquPKK8nwnUJJXD_AP/view?usp=sharing)  | [DR_OQ](https://drive.google.com/file/d/18qBHeSYlKh9RRv4xuI79NGS1-XHbcvrr/view?usp=sharing) |
    | [DR_QV](https://drive.google.com/file/d/12SZLuI4ApLEagqBF7zY-SYh7WqJ8QAy3/view?usp=sharing)  | [DR_QV](https://drive.google.com/file/d/13Ptr4hiy7tjuwiC3aK0dq29EL4oQD_Sy/view?usp=sharing) |
    | [DR_OQ->QV](https://drive.google.com/file/d/1pRINHVP566LTJp5XLr4R4M1UYiF9k_Dz/view?usp=sharing)  | [DR_OQ->QV](https://drive.google.com/file/d/1lMFDSZeiuW75BCNWbdX4bo-WCrkRpfyk/view?usp=sharing) |
    | [RoDR w/ DR_OQ](https://drive.google.com/file/d/1cW7g25y7eWg-rqlcLzZj141fiHqbnkUe/view?usp=sharing)  | [RoDR w/ DR_OQ](https://drive.google.com/file/d/1O7HRb-DU5RV-UjVl2qdL9MCfuo0iiATD/view?usp=sharing) |

3. Retrieval files<sup>*</sup>:

    | Dataset | DR_OQ | RoDR w/ DR_OQ | 
    |----------|-----|-----|
    | Passage-Dev | [Download](https://drive.google.com/file/d/16Ic9-FloPDUvlpAl-euNdmDznLCLAqh2/view?usp=sharing) | [Download](https://drive.google.com/file/d/1gFnUwfvcYgvBIwvpxcm2W6Ge6lbU6NcA/view?usp=sharing) |
    | Document-Dev | [Download](https://drive.google.com/file/d/1kyf1t96k7UjJXD16Jj1hroC7TP75_u0S/view?usp=sharing) | [Download](https://drive.google.com/file/d/1E1KOrbsGXFpyhKgCwyVHyNlYGRQVOwlg/view?usp=sharing) |
    | ANTIQUE | [Download](https://drive.google.com/file/d/1NbCn31bRu0oSACrqOWG6pBFI2qCiK56z/view?usp=sharing) | [Download](https://drive.google.com/file/d/13wpHh_Hsu0tQCm4OzPls9qqfjC72VB2u/view?usp=sharing) |
    
    <sup>*</sup> Due to the large size of run files on Passage-Dev, we only provide the run files of 
    `DR_OQ` and `RoDR w/ DR_OQ` models. If you want to obtain the run files of `DR_QV` and `DR_OQ->QV` 
    models, please feel free to contact us. 

## RoDR on existing DR models
If you want to apply RoDR to publicly available DR models, such as ANCE, TAS-Balanced and ADORE+STAR, which are enhanced
in our paper, you need to make some minor changes in the model level, such as adding the pooler in ANCE, and using 
separate query and passage encoders in ADORE+STAR.
Herein, we provide the model checkpoints and retrieval files for the reproducibility of our experiments and other research uses.
1. Models:

    | Original | RoDR |
    |-----|-----|
    | [ANCE](https://drive.google.com/file/d/1tsqT5oCsnKCcQASTuPT1Qzl1RifygxlA/view?usp=sharing) | [RoDR w/ ANCE](https://drive.google.com/file/d/1CuFZcZOk2_1ZmNz728SNXnf0l3G29lGm/view?usp=sharing) |
    | [TAS-Balanced](https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco) | [RoDR w/ TAS-Balanced](https://drive.google.com/file/d/1MK3baDlfS0ypj_mW5ySMCBjCNhBiYNVj/view?usp=sharing) |
    | [ADORE+STAR](https://drive.google.com/file/d/1BQKpxUNnb8bSXLGQBemEOk5c_X3vyZPh/view?usp=sharing) | [RoDR w/ ADORE+STAR](https://drive.google.com/file/d/1JkguYtan1N-XTtYUtK1iX-_fXtgsU1O5/view?usp=sharing) |

2. Retrieval files<sup>**</sup>:

    | Model | Passage-Dev | ANTIQUE |
    |----------|-----|-----|
    | RoDR w/ ANCE | [Download](https://drive.google.com/file/d/1zfc7ss4MHqAX7-y-8ZV7dy3ZJoIwC9FH/view?usp=sharing) | [Download](https://drive.google.com/file/d/171zLsLGqUeQxWa-eBK76QuOBk0ZTpUzE/view?usp=sharing) |
    | RoDR w/ TAS-Balanced | [Download](https://drive.google.com/file/d/1Aq25MQv1YQqOSm4GOybfJuCgDHLyBOI-/view?usp=sharing) | [Download](https://drive.google.com/file/d/1gIu5AcmtPBdjvfu7yUQ8WGzlk0dyTAVv/view?usp=sharing) |
    | RoDR w/ ADORE+STAR | [Download](https://drive.google.com/file/d/16bUYB91gSlPDH1sdNTYI75WkVgmYgyXP/view?usp=sharing) | [Download](https://drive.google.com/file/d/1884f88D-58JJGZQ3j8BNeeh-akXaH1w8/view?usp=sharing) |
    
    <sup>**</sup> Due to the large size of run files on Passage-Dev, we only provide the run files of RoDR models.
    If you want to obtain the run files of original existing DR models, please feel free to contact us. 
    
## Citation
If you find our paper/resources useful, please cite:
```
@inproceedings{chen_ijcai2022-275,
  title     = {Towards Robust Dense Retrieval via Local Ranking Alignment},
  author    = {Xuanang Chen and
               Jian Luo and
               Ben He and
               Le Sun and
               Yingfei Sun},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {1980--1986},
  year      = {2022}
}
```
