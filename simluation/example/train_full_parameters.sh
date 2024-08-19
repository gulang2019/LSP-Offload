DATA_PATH="theblackcat102/evol-codealpaca-v1"
OUTPUT_PATH="./outputs/full_parameters"
MODEL="deepseek-ai/deepseek-coder-1.3b-base"

lr=$1
wrs=$2

python finetune_deepseekcoder.py \
    --model_name_or_path $MODEL \
    --data_path $DATA_PATH \
    --output_dir "${OUTPUT_PATH}_${lr}_wrs${wrs}" \
    --num_train_epochs 5 \
    --model_max_length 1024 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate $lr \
    --warmup_steps ${wrs} \
    --logging_steps 10 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --bf16 True
    # --gradient_checkpointing True \