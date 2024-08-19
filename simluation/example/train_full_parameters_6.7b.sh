DATA_PATH="theblackcat102/evol-codealpaca-v1"
OUTPUT_PATH="./outputs/full_parameters_6.7b"
MODEL="deepseek-ai/deepseek-coder-6.7b-base"

lr=$1
lr_min=$2
wrs=$3

accelerate launch finetune_deepseekcoder.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --output_dir "${OUTPUT_PATH}_${lr}_wrs${wrs}_2" \
    --num_train_epochs 1 \
    --model_max_length 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 348 \
    --learning_rate $lr \
    --warmup_steps ${wrs} \
    --logging_steps 10 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": '$lr_min'}' \
    --report_to "tensorboard" \
    --bf16 True
    # --gradient_checkpointing True \