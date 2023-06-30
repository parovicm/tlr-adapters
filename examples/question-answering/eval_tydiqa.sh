#!/bin/bash

# Evaluate on tydiqa dataset

python run_qa.py \
    --model_name_or_path bert-base-multilingual-cased \
    --dataset_name xtreme \
    --dataset_config_name tydiqa \
    --data_language ${LANG_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --do_eval \
    --per_device_eval_batch_size 16 \
    --overwrite_output_dir \
    --train_adapter \
    --load_adapter "$TASK_ADAPTER/xtreme" \
    --task_adapter_drop_last yes \
    --load_lang_adapter ${LANG_ADAPTER} \
    --lang_adapter_drop_last yes
