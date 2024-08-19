paths = [
    ('LSP-Offload (S=256, d=16)', 'llama-3b-lsp-256-16', 1.77),
    ('LSP-Offload (S=512, d=16)', 'llama-3b-lsp-512-16', 1.80),
    ('LSP-Offload (S=1024, d=16)', 'llama-3b-lsp-1024-16', 1.94),
    ('LoRA (Rank=16)', 'llama-3b-lora', 1.79),
    ('Zero-Offload', 'llama-3b-zero', 3.63)
]
iter_per_epoch = 11000

import matplotlib.pyplot as plt
import argparse
import pandas as pd 

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', type = str, default = 'loss')
parser.add_argument('--hours', type = int, default = 1000)
parser.add_argument('--rolling', type = int, default = 1)
parser.add_argument('--min-periods', type = int, default = 1)
args = parser.parse_args()

fig, ax = plt.subplots(figsize = (8,8))
for name, path, time_per_iter in paths:
    df = pd.read_csv(f'result/{path}/eval_losses.csv')
    df.loc[0, 'ppl'] = 2.5
    df['Elapsed'] = (df['epoch'] * iter_per_epoch + df['iter']) * time_per_iter
    df['Hours'] = df['Elapsed'] / 3600
    df = df[df['Hours'] <= args.hours]

    # Calculate the rolling mean and standard deviation of 'ppl'
    df['ppl_mean'] = df['ppl'].rolling(args.rolling, min_periods=args.min_periods).mean()
    df['ppl_std'] = df['ppl'].rolling(args.rolling, min_periods=args.min_periods).std() * 0.1

    # Fill the area between the line minus the standard deviation and the line plus the standard deviation
    ax.fill_between(df['Hours'], df['ppl_mean'] - df['ppl_std'], df['ppl_mean'] + df['ppl_std'], alpha=0.2)

    # Plot the rolling mean of 'ppl'
    ax.plot(df['Hours'], df['ppl_mean'], label=name, linewidth=3)
    ax.set_xlabel('Time (h)', fontsize = 30)
    ax.set_ylabel('Perplexity', fontsize = 30)
plt.xticks(fontsize = 25)
plt.yticks(fontsize = 25)
print(f'eval result saved to {args.output}_eval.png')
plt.legend(fontsize = 20)
fig.savefig(f'{args.output}_eval.png', bbox_inches = 'tight')
fig.savefig(f'{args.output}_eval.pdf', bbox_inches = 'tight')


fig, ax = plt.subplots()
for name, path, time_per_iter in paths:
    df = pd.read_csv(f'result/{path}/losses.csv')
    df['Elapsed'] = (df['epoch'] * iter_per_epoch + df['iter']) * time_per_iter
    df['Hours'] = df['Elapsed'] / 3600
    df = df[df['Hours'] <= args.hours]

    # Calculate the rolling mean and standard deviation of 'ppl'
    df['loss_mean'] = df['loss'].rolling(args.rolling, min_periods = args.min_periods).mean()
    df['loss_std'] = df['loss'].rolling(args.rolling, min_periods = args.min_periods).std()

    # Fill the area between the line minus the standard deviation and the line plus the standard deviation
    ax.fill_between(df['Hours'], df['loss_mean'] - df['loss_std'], df['loss_mean'] + df['loss_std'], alpha=0.2)

    # Plot the rolling mean of 'loss'
    ax.plot(df['Hours'], df['loss_mean'], label=name)

ax.legend()
print(f'training saved to {args.output}_train.png')
fig.savefig(f'{args.output}_train.png')
fig.savefig(f'{args.output}_train.pdf')