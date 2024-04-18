#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

MODEL="/mnt/d/LLM/models/Qwen/Qwen-VL-Chat" # Qwen/Qwen-VL-Chat-Int4 Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="train"

export CUDA_VISIBLE_DEVICES=0

python finetune.py \
    --model_name_or_path $MODEL \
    --data_path $DATA/train.json \
    --bf16 True \
    --fix_vit True \
    --output_dir $DATA/output \
    --cache_dir $DATA/cache \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 1024 \
    --lazy_preprocess True \
    --gradient_checkpointing \
    --use_lora 

if [ $? -eq 0 ]; then
    DATESTR=`date +%Y%m%d-%H%M%S`
    OUTPUT=./train/output-$DATESTR-lora
    rsync -a --exclude='*/' ./train/output/ $OUTPUT
    echo "python3 openai_api.py --server-name 0.0.0.0 --checkpoint-path $OUTPUT" > ./api.sh
    ./api.sh
fi
