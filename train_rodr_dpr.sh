#!/bin/bash

pd_tbs=16
lr=5e-6
epoch=4
save_steps=50000
q_max_len=32
p_max_len=128
w1=1.0
w2=0.0
w3=0.2
gpu_id=1

train_data_name=rodr_oq_128l_30n
train_data_dir=./msmarco_passage/process/train_data/${train_data_name}
model_dir=./msmarco_passage/models/oq_128l_30n/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}
output_dir=./msmarco_passage/models/${train_data_name}/bert_base_q${q_max_len}p${p_max_len}_pdbs${pd_tbs}_lr${lr}_ep${epoch}_w1${w1}_w2${w2}_w3${w3}

CUDA_VISIBLE_DEVICES=${gpu_id} python -m tevatron.driver.train \
                                  --training_mode oq.qv.lra \
                                  --output_dir ${output_dir} \
                                  --model_name_or_path ${model_dir} \
                                  --save_steps ${save_steps} \
                                  --logging_steps 500 \
                                  --train_dir ${train_data_dir} \
                                  --fp16 \
                                  --per_device_train_batch_size ${pd_tbs} \
                                  --learning_rate ${lr} \
                                  --num_train_epochs ${epoch} \
                                  --dataloader_num_workers 2 \
                                  --train_n_passages 8 \
                                  --q_max_len ${q_max_len} \
                                  --p_max_len ${p_max_len} \
                                  --w1 ${w1} \
                                  --w2 ${w2} \
                                  --w3 ${w3}