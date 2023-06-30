#!/bin/bash

# Train MAD-X style task adapter on SQuAD dataset
# LANG_ADAPTER is a path to the English language adapter

python run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --dataset_name xtreme \
  --dataset_config_name SQuAD \
  --language en \
  --output_dir ${OUTPUT_DIR} \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 15 \
  --save_steps 625 \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config pfeiffer \
  --task_adapter_drop_last yes \
  --adapter_reduction_factor 16 \
  --load_lang_adapter ${LANG_ADAPTER} \
  --lang_adapter_drop_last yes \
  --learning_rate 1e-4 \
  --evaluation_strategy steps \
  --eval_steps 625 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --patience 4
