#!/bin/bash

# Train target task adapter on the NLI dataset
# LANG_ADAPTER is path to the target language adapter

python run_nli.py \
  --model_name_or_path bert-base-multilingual-cased \
  --dataset_name multi_nli \
  --language en \
  --output_dir ${OUTPUT_DIR} \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --num_train_epochs 5 \
  --save_steps 625 \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config pfeiffer \
  --task_adapter_drop_last yes \
  --adapter_reduction_factor 16 \
  --load_lang_adapter ${LANG_ADAPTER} \
  --lang_adapter_drop_last yes \
  --learning_rate 2e-5 \
  --evaluation_strategy steps \
  --eval_steps 625 \
  --metric_for_best_model eval_accuracy \
  --load_best_model_at_end \
  --eval_split validation_matched
