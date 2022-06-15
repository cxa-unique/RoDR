#!/bin/bash

pd_tbs=16
lr=5e-6
epoch=4
q_max_len=32
p_max_len=128

model_name=oq_128l_30n
model_dir=./msmarco_passage/models/${model_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}
save_path=./msmarco_passage/results/${model_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}
gpu_id=1

### encode corpus and queries
CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
                                                         --model_name_or_path ${model_dir} \
                                                         --fp16 \
                                                         --p_max_len 128 \
                                                         --per_device_eval_batch_size 128 \
                                                         --encode_in_path ./msmarco_passage/process/corpus_128/corpus.128.json \
                                                         --encoded_save_path ${save_path}/corpus_128.pt

query_dir=./msmarco_passage/process/query/dev
CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
                                                       --model_name_or_path ${model_dir} \
                                                       --fp16 \
                                                       --q_max_len 32 \
                                                       --encode_is_qry \
                                                       --per_device_eval_batch_size 128 \
                                                       --encode_in_path ${query_dir}/queries.dev.small.32.json \
                                                       --encoded_save_path ${save_path}/queries.dev.small.32.pt

VARIATION_TYPES=(MisSpell ExtraPunc BackTrans SwapSyn_Glove SwapSyn_WNet TransTense NoStopword SwapWords)
for qv_type in ${VARIATION_TYPES[@]}
do
CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
                                                       --model_name_or_path ${model_dir} \
                                                       --fp16 \
                                                       --q_max_len 32 \
                                                       --encode_is_qry \
                                                       --per_device_eval_batch_size 128 \
                                                       --encode_in_path ${query_dir}/queries.dev.small.${qv_type}.32.json \
                                                       --encoded_save_path ${save_path}/queries.dev.small.${qv_type}.32.pt
done


## index and retrieval
index_type=Flat
index_dir=${save_path}/${index_type}

python -m tevatron.faiss_retriever \
          --query_reps ${save_path}/queries.dev.small.32.pt \
          --passage_reps ${save_path}/corpus_128.pt \
          --index_type ${index_type} \
          --batch_size 256 \
          --depth 1000 \
          --save_ranking_file ${index_dir}/queries.dev.small.q32.p128.top1k.run.txt \
          --save_index \
          --save_index_dir ${index_dir}

VARIATION_TYPES=(MisSpell ExtraPunc BackTrans SwapSyn_Glove SwapSyn_WNet TransTense NoStopword SwapWords)
for qv_type in ${VARIATION_TYPES[@]}
do
  python -m tevatron.faiss_retriever \
            --query_reps ${save_path}/queries.dev.small.${qv_type}.32.pt \
            --passage_index ${index_dir}/${index_type}.index \
            --passage_lookup ${index_dir}/p_lookup.pt\
            --batch_size 256 \
            --depth 1000 \
            --save_ranking_file ${index_dir}/queries.dev.small.${qv_type}.q32.p128.top1k.run.txt
done