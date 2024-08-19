#!/bin/bash
DATA_PATH="theblackcat102/evol-codealpaca-v1"
OUTPUT_PATH="./test${3}_${4}_${1}_wrs${2}"
MODEL="deepseek-ai/deepseek-coder-1.3b-base"

python ./sft_main_lsp.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --optim adamw_lsp \
    --model_max_length 1024 \
    --train_batch_size 16 \
    --num_dataloader_workers 4 \
    --gradient_accumulation_steps 8 \
    --lr $1 \
    --max_epochs 5 \
    --lr_scheduler_type cosine \
    --warmup_steps $2 \
    --optim_target_modules "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj" \
    --compressor_name count_sketch \
    --compressor_update_freq 100 \
    --compressor_update_num_samples 128 \
    --output_dir $OUTPUT_PATH \
    --optim_args "cs_lr_scheduler_type=cosine,cs_optimizer_type=adamw,init_method=binary,nonzero_dim=${4},output_dim=${3},optimize_lr=${5},common_optimize_iter=100,specific_optimize_iter=0" \
    --compressor_layerwise
