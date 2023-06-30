#!/bin/bash

# Evaluate on XNLI dataset

python run_nli.py \
    --model_name_or_path bert-base-multilingual-cased \
    --dataset_name xnli \
    --dataset_config_name $LANG \
    --output_dir ${OUTPUT_DIR} \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --overwrite_output_dir \
    --train_adapter \
    --load_adapter "$TASK_ADAPTER/nli" \
    --task_adapter_drop_last yes \
    --load_lang_adapter ${LANG_ADAPTER} \
    --eval_split test \
    --lang_adapter_drop_last yes
