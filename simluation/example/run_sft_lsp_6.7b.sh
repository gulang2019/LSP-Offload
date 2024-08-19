#!/bin/bash
DATA_PATH="theblackcat102/evol-codealpaca-v1"
OUTPUT_PATH="./test_lsp_6.7B"
MODEL="deepseek-ai/deepseek-coder-6.7b-base"

python ./sft_main_lsp.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --optim adamw_lsp \
    --model_max_length 1024 \
    --train_batch_size 4 \
    --num_dataloader_workers 4 \
    --gradient_accumulation_steps 16 \
    --lr $1 \
    --lr_min $2 \
    --max_epochs 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --warmup_steps $3 \
    --optim_target_modules "q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj" \
    --compressor_name count_sketch \
    --compressor_update_freq 100 \
    --compressor_update_num_samples 128 \
    --output_dir $OUTPUT_PATH \
    --optim_args "cs_lr_scheduler_type=cosine,cs_optimizer_type=adamw,init_method=binary,nonzero_dim=$5,output_dim=$4,optimize_lr=$6,common_optimize_iter=100,specific_optimize_iter=0" \
    --compressor_layerwise
