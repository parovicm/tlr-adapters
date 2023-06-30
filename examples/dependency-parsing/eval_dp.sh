#!/bin/bash

# Evaluate on UD dataset

python run_udp.py \
    --model_name_or_path ${BASE_MODEL} \
    --dataset_name universal_dependencies \
    --dataset_config_name "${LANG}_${TREEBANK}" \
    --output_dir ${OUTPUT_DIR} \
    --do_eval \
    --per_device_eval_batch_size 8 \
    --overwrite_output_dir \
    --train_adapter \
    --load_adapter "$TASK_ADAPTER/udp" \
    --task_adapter_drop_last yes \
    --load_lang_adapter ${LANG_ADAPTER} \
    --lang_adapter_drop_last yes \
    --eval_split ${SPLIT}
