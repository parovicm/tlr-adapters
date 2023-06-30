#!/bin/bash

# Train bilingual task-adapter on the NER dataset
# File passed with --lang_adapters_file needs to contains paths to English and target language adapters

python run_ner.py \
  --model_name_or_path bert-base-multilingual-cased \
  --dataset_name ner_dataset.py \
  --dataset_config_name en \
  --output_dir ${OUTPUT_DIR} \
  --do_train \
  --do_eval \
  --label_column_name ner_tags \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 10 \
  --save_steps 250 \
  --task_name ner \
  --overwrite_output_dir \
  --train_adapter \
  --adapter_config pfeiffer \
  --task_adapter_drop_last yes \
  --adapter_reduction_factor 16 \
  --load_lang_adapter ${LANG_ADAPTER} \
  --lang_adapter_drop_last yes \
  --learning_rate 5e-5 \
  --evaluation_strategy steps \
  --eval_steps 250 \
  --metric_for_best_model eval_f1 \
  --load_best_model_at_end \
  --eval_split validation \
  --lang_adapters_file ${ADAPTERS_FILE}
