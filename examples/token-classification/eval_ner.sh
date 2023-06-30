#!/bin/bash

# Evaluate on the MasakhaNER dataset

python run_ner.py \
      --model_name_or_path bert-base-multilingual-cased \
      --dataset_name ner_dataset.py \
      --dataset_config_name ${LANG} \
      --output_dir ${OUTPUT_DIR} \
      --do_eval \
      --label_column_name ner_tags \
      --per_device_eval_batch_size 8 \
      --task_name ner \
      --overwrite_output_dir \
      --train_adapter \
      --load_adapter "$TASK_ADAPTER/ner" \
      --task_adapter_drop_last yes \
      --load_lang_adapter ${LANG_ADAPTER} \
      --lang_adapter_drop_last yes \
      --eval_split test
