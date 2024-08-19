mkdir -p result
# LoRA
python ./llm_finetuning.py\
    --task alpaca --model llama-3b\
    --lr 1e-5\
    --modules attn.qkv attn.o mlp.up mlp.down\
    --max_seqlen 512\
    --dtype bfloat16\
    --batch-size 16\
    --optim-dtype float32\
    --eval_freq 1000\
    --save\
    --peft lora\
    --rank 8\
    --output result/llama-3b-lora

# LSP 1024 16
python ./llm_finetuning.py \
    --task alpaca \
    --model llama-3b\
    --lr 1e-4\
    --modules attn.qkv attn.o mlp.up mlp.down\
    --max_seqlen 512\
    --dtype bfloat16\
    --batch-size 16\
    --optim-dtype float32\
    --eval_freq 1000\
    --save\
    --eval_freq 1000\
    --offload cpu\
    --compress-profile 1000\
    --compress CS\
    --compress-profile-data 16\
    --compress-cs-size 1024\
    --compress-cs-n_unempty 16\
    --compress-cs-n_iter 100\
    --compress-cs-n_ft_iter 0\
    --compress-cs-lr 1e-1\
    --compress-cs-ft_lr 1e-1\
    --save\
    --stop 5000\
    --gradient-checkpoint\
    --output result/llama-3b-lsp-1024-16

# LSP 512 16
python ./llm_finetuning.py \
    --task alpaca \
    --model llama-3b\
    --lr 1e-4\
    --modules attn.qkv attn.o mlp.up mlp.down\
    --max_seqlen 512\
    --dtype bfloat16\
    --batch-size 16\
    --optim-dtype float32\
    --eval_freq 1000\
    --offload cpu\
    --compress-profile 1000\
    --compress CS\
    --compress-profile-data 16\
    --compress-cs-size 512\
    --compress-cs-n_unempty 16\
    --compress-cs-n_iter 100\
    --compress-cs-n_ft_iter 0\
    --compress-cs-lr 1e-1\
    --compress-cs-ft_lr 1e-1\
    --save\
    --stop 5000\
    --gradient-checkpoint\
    --output result/llama-3b-lsp-512-16

# LSP 256 16
python ./llm_finetuning.py \
    --task alpaca \
    --model llama-3b\
    --lr 1e-4\
    --modules attn.qkv attn.o mlp.up mlp.down\
    --max_seqlen 512\
    --dtype bfloat16\
    --batch-size 16\
    --optim-dtype float32\
    --eval_freq 1000\
    --save\
    --offload cpu\
    --compress-profile 1000\
    --compress CS\
    --compress-profile-data 16\
    --compress-cs-size 256\
    --compress-cs-n_unempty 16\
    --compress-cs-n_iter 100\
    --compress-cs-n_ft_iter 0\
    --compress-cs-lr 1e-1\
    --compress-cs-ft_lr 1e-1\
    --save\
    --stop 5000\
    --gradient-checkpoint\
    --output result/llama-3b-lsp-256-16
# Zero
python ./llm_finetuning.py \
    --task alpaca \
    --model llama-3b\
    --lr 1e-5\
    --modules attn.qkv attn.o mlp.up mlp.down\
    --max_seqlen 512\
    --dtype bfloat16\
    --batch-size 16\
    --optim-dtype float32\
    --eval_freq 1000\
    --save --max_seqlen 512\
    --optim-dtype float32\
    --eval_freq 1000\
    --offload cpu\
    --compress-profile 1000\
    --save\
    --stop 5000\
    --gradient-checkpoint\
    --output result/llama-3b-zero

mkdir -p pics

python scripts/draw_loss_llama_3b.py --rolling 20 --output pics/llama-3b