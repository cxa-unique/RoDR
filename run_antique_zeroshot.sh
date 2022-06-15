#!/bin/bash

pd_tbs=16
lr=5e-6
epoch=4
q_max_len=32
p_max_len=128

model_name=rodr_oq_128l_30n
model_dir=./msmarco_passage/models/${model_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}_w11.0_w21.0_w30.2
save_path=./antique/marco_pass_zeroshot_results/${model_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}_w11.0_w21.0_w30.2
gpu_id=1

#### ANTIQUE
CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
                                                         --model_name_or_path ${model_dir} \
                                                         --fp16 \
                                                         --p_max_len 128 \
                                                         --per_device_eval_batch_size 128 \
                                                         --encode_in_path ./antique/corpus_128/corpus.128.json \
                                                         --encoded_save_path ${save_path}/corpus_128/corpus_128.pt

CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
                                                     --model_name_or_path ${model_dir} \
                                                     --fp16 \
                                                     --q_max_len 32 \
                                                     --encode_is_qry \
                                                     --per_device_eval_batch_size 128 \
                                                     --encode_in_path ./antique/query/antique.train.split200-valid.queries.32.json \
                                                     --encoded_save_path ${save_path}/antique.train.split200-valid.queries.32.pt

index_type=Flat
index_dir=${save_path}/${index_type}

python -m tevatron.faiss_retriever \
          --query_reps ${save_path}/antique.train.split200-valid.queries.32.pt \
          --passage_reps ${save_path}/corpus_128/corpus_128.pt \
          --index_type ${index_type} \
          --batch_size 256 \
          --depth 1000 \
          --save_ranking_file ${index_dir}/antique.train.split200-valid.q32.p128.top1k.run.txt \
          --save_index \
          --save_index_dir ${index_dir}

qv_types=(T5DescToTitle.136 T5QQP.105 BackTransl.93 WEmbedSynSwap.124 WNetSynSwap.72)
for qv_type in ${qv_types[@]}
do
  CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.encode --output_dir=temp \
                                                       --model_name_or_path ${model_dir} \
                                                       --fp16 \
                                                       --q_max_len 32 \
                                                       --encode_is_qry \
                                                       --per_device_eval_batch_size 128 \
                                                       --encode_in_path ./antique/query/antique.${qv_type}.query.32.json \
                                                       --encoded_save_path ${save_path}/antique.${qv_type}.query.32.pt

  python -m tevatron.faiss_retriever \
              --query_reps ${save_path}/antique.${qv_type}.query.32.pt \
              --passage_index ${index_dir}/${index_type}.index \
              --passage_lookup ${index_dir}/p_lookup.pt\
              --batch_size 256 \
              --depth 1000 \
              --save_ranking_file ${index_dir}/antique.${qv_type}.q32.p128.top1k.run.txt
done