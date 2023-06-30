#!/bin/bash

# Train target task adapter on the UD dataset
# LANG_ADAPTER is a path to the target language adapter

python run_udp.py \
  --model_name_or_path bert-base-multilingual-cased \
  --dataset_name universal_dependencies \
  --dataset_config_name en_ewt \
  --output_dir ${OUTPUT_DIR} \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 10 \
  --save_steps 250 \
  --train_adapter \
  --adapter_config pfeiffer \
  --task_adapter_drop_last yes \
  --adapter_reduction_factor 16 \
  --load_lang_adapter ${LANG_ADAPTER} \
  --lang_adapter_drop_last yes \
  --learning_rate 5e-5 \
  --evaluation_strategy steps \
  --eval_steps 250 \
  --eval_split validation \
  --metric_for_best_model eval_las \
  --load_best_model_at_end \
  --overwrite_output_dir
