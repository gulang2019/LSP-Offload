## Experiment Environment
1. The laptop: Nvidia A1000 Laptop GPU (4GB) and Intel Core-i7 12800H CPU (32GB)
2. The workstation: Nvidia RTX 4090 GPU (24 GB) and AMD Ryzen Threadripper 3970X CPU (252GB)

## Setup 
```
mkdir -p result
mkdir -p datasets
wget -O datasets/alpaca_gpt4_data.json https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data.json
cd kernels
DEVICE=cpu python setup.py build_ext --inplace 2>&1 | tee err.txt
cd ..
```

## Reproduce Experiments
### Table 3
```bash
source scripts/run_glue.sh
```

See evaluation losses in result/roberta-base-LSP-$task/eval_losses.csv.
See evaluation losses of full parameter training in result/roberta-base-FullParam-$task/eval_losses.csv.

### Figure 5a
We conduct this experiment on the laptop GPU.
```bash
source scripts/run-gpt2-large.sh
```
### Figure 5b
We conduct this experiment on the workstation GPU.
```
source scripts/run-llama-3b.sh
``` 

### Figure 5c


1. Run End2End training;
**TODO:Zhuofeng, add script here**

2. Profile the per-iteration time;
We conduct this experiment on the laptop GPU.
```bash
# For LSP-Offload
python para_offload_modules.py --offload cpu --gradient-checkpointing  --model deepseek-ai/deepseek-coder-1.3b-base --compress-size 1280 --nnz 4  --sch-fcfs-process-delay 2 --sch-fcfs-h2d-delay 3 --sch-fcfs-upd-delay 4  --sch-fcfs-point 40 --sch-lcfs-d2h-delay 1000 --seq_len 384 --n_repeat 10 --compress --bs 1
# For Zero
python para_offload_modules.py --offload cpu --gradient-checkpointing  --model deepseek-ai/deepseek-coder-1.3b-base --n_repeat 10 --zero --bs 1
```
3. combine first 2 steps into time v.s. loss csvs (manual done) and plot the figures
```bash
python scripts/draw_loss_code_1.3B.py --output pics/code_1.3B --rolling 20 --hours 120
```
### Figure 5d, Table 4
1. Run End2End training;
**TODO:Zhuofeng, add script here**

2. Profile the per-iteration time;
We conduct this experiment on the workstation GPU.
```bash
# For LSP-Offload
python para_offload_modules.py --offload cpu  --model deepseek-ai/deepseek-coder-6.7b-base --compress-size 2048 --nnz 8  --sch-fcfs-process-delay 1 --sch-fcfs-h2d-delay 2 --sch-fcfs-upd-delay 2  --sch-fcfs-point 10 --seq_len 1024 --n_repeat 10 --compress --bs 4 --gradient-checkpoint
# For Zero
python para_offload_modules.py --offload cpu  --model deepseek-ai/deepseek-coder-6.7b-base --zero --bs 4 --gradient-checkpoint
```
3. combine first 2 steps into time v.s. loss csvs (manual done) and plot the figures
```bash
python scripts/draw_loss_code_6.7B.py --output pics/code_6.7B --rolling 20
```