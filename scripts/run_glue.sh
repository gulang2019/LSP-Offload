# Ours
export CUDA_VISIBLE_DEVICES=0
# for task in mnli sst2 mrpc cola qnli qqp stsb
for task in mnli
do  
    python ./llm_finetuning.py \
        --task $task \
        --save \
        --modules attn.qkv attn.o mlp.gate mlp.up mlp.down  \
        --model roberta-base \
        --max_seqlen 128 \
        --dtype bfloat16 \
        --batch-size 16 \
        --optim-dtype float32 \
        --eval_freq 100 \
        --offload cuda:0 \
        --compress-cs-size 512 \
        --compress-profile 1000 \
        --compress-profile-data 16 \
        --compress-cs-n_iter 10 \
        --compress-cs-n_ft_iter 100 \
        --compress-cs-n_unempty 16 \
        --compress CS\
        --n_epoch 30 \
        --gradient-checkpointing \
        --sch-fcfs-point 140 \
        --sch-fcfs-process-delay 8 \
        --sch-fcfs-h2d-delay 8 \
        --sch-lcfs-d2h-delay 1000 \
        --sch-lcfs-process-delay 8 \
        --sch-lcfs-h2d-delay 8 \
        --lr 5e-5 \
        --compress-cs-thresh 0.3 \
        --timeout 1 \
        --output result/roberta-base-LSP-$task
done

for task in mnli
do  
    python ./llm_finetuning.py \
        --task $task \
        --save \
        --modules attn.qkv attn.o mlp.gate mlp.up mlp.down  \
        --model roberta-base \
        --max_seqlen 128 \
        --dtype bfloat16 \
        --batch-size 16 \
        --optim-dtype float32 \
        --eval_freq 100 \
        --offload cuda:0 \
        --n_epoch 30 \
        --gradient-checkpointing \
        --sch-fcfs-point 140 \
        --sch-fcfs-process-delay 8 \
        --sch-fcfs-h2d-delay 8 \
        --sch-lcfs-d2h-delay 1000 \
        --sch-lcfs-process-delay 8 \
        --sch-lcfs-h2d-delay 8 \
        --lr 5e-5 \
        --compress-cs-thresh 0.3 \
        --timeout 1 \
        --output result/roberta-base-FullParam-$task
done