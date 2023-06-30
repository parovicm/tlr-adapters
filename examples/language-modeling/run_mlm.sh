#!/bin/bash

python run_mlm.py \
  --language $LANG \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file ${DATA_FILE} \
  --output_dir ${OUTPUT_DIR} \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --max_seq_length 256 \
  --learning_rate 5e-5 \
  --min_steps 30000 \
  --max_steps 100000 \
  --num_train_epochs 100 \
  --save_steps 1000 \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config pfeiffer+inv \
  --adapter_reduction_factor 2 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --validation_split_percentage 5 \
  --load_best_model_at_end
