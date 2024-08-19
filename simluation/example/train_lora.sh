DATA_PATH="theblackcat102/evol-codealpaca-v1"
OUTPUT_PATH="./outputs/lora"
MODEL="deepseek-ai/deepseek-coder-1.3b-base"

lr=$1
warmup_steps=$2
lora_rank=$3
lora_alpha=$4

python finetune_deepseekcoder_lora.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --output_dir "${OUTPUT_PATH}_${lr}_wrs${warmup_steps}_2" \
    --num_train_epochs 5 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate $lr \
    --warmup_steps $warmup_steps \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --bf16 True \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    # --gradient_checkpointing True \