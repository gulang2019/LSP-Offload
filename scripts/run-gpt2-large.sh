mkdir -p result
# LSP 512 8
python ./llm_finetuning.py \
    --task alpaca \
    --model gpt2-large \
    --lr 1e-4 \
    --modules attn.qkv attn.o mlp.up mlp.down \
    --max_seqlen 512 \
    --dtype bfloat16 \
    --batch-size 4 \
    --optim-dtype float32 \
    --eval_freq 1000 \
    --offload cpu  \
    --compress-profile 1000 \
    --compress CS \
    --compress-profile-data 16 \
    --compress-cs-size 512 \
    --compress-cs-n_unempty 8 \
    --compress-cs-n_iter 100 \
    --compress-cs-n_ft_iter 0 \
    --compress-cs-lr 1e-1 \
    --compress-cs-ft_lr 1e-1 \
    --save \
    --stop 5000 \
    --gradient-checkpoint\
    --output result/gpt2-large-lsp-512-8

# LSP 512 16
python ./llm_finetuning.py \
    --task alpaca \
    --model gpt2-large \
    --lr 1e-4 \
    --modules attn.qkv attn.o mlp.up mlp.down \
    --max_seqlen 512 \
    --dtype bfloat16 \
    --batch-size 4 \
    --optim-dtype float32 \
    --eval_freq 1000 \
    --offload cpu  \
    --compress-profile 1000 \
    --compress CS \
    --compress-profile-data 16 \
    --compress-cs-size 512 \
    --compress-cs-n_unempty 16 \
    --compress-cs-n_iter 100 \
    --compress-cs-n_ft_iter 0 \
    --compress-cs-lr 1e-1 \
    --compress-cs-ft_lr 1e-1 \
    --save \
    --stop 5000 \
    --gradient-checkpoint\
    --output result/gpt2-large-lsp-512-16

# LSP 512 32
python ./llm_finetuning.py \
    --task alpaca \
    --model gpt2-large \
    --lr 1e-4 \
    --modules attn.qkv attn.o mlp.up mlp.down \
    --max_seqlen 512 \
    --dtype bfloat16 \
    --batch-size 4 \
    --optim-dtype float32 \
    --eval_freq 1000 \
    --offload cpu  \
    --compress-profile 1000 \
    --compress CS \
    --compress-profile-data 16 \
    --compress-cs-size 512 \
    --compress-cs-n_unempty 32 \
    --compress-cs-n_iter 100 \
    --compress-cs-n_ft_iter 0 \
    --compress-cs-lr 1e-1 \
    --compress-cs-ft_lr 1e-1 \
    --save \
    --stop 5000 \
    --gradient-checkpoint\
    --output result/gpt2-large-lsp-512-32

# zero offload
python ./llm_finetuning.py \
    --task alpaca \
    --model gpt2-large \
    --lr 1e-4 \
    --modules attn.qkv attn.o mlp.up mlp.down \
    --max_seqlen 512 \
    --dtype bfloat16 \
    --batch-size 4 \
    --optim-dtype float32 \
    --eval_freq 1000 \
    --offload cpu \
    --save \
    --stop 5000 \
    --gradient-checkpoint \
    --output result/gpt2-large-zero-offload

#lora
python ./llm_finetuning.py \
    --task alpaca \
    --model gpt2-large \
    --lr 1e-5 \
    --modules attn.qkv attn.o mlp.up mlp.down \
    --max_seqlen 512 \
    --dtype bfloat16 \
    --batch-size 4 \
    --optim-dtype float32 \
    --eval_freq 1000 \
    --save \
    --peft lora \
    --rank 32 \
    --stop 5000 \
    --gradient-checkpoint\
    --output result/gpt2-large-lora

# visualize
mkdir -p pics
python scripts/draw_loss_gpt2_large.py --rolling 20 --output pics/gpt2-large --hours 15