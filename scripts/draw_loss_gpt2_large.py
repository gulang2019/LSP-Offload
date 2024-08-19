paths = [
    ('LSP-Offload (d=512, r = 16)', 'gpt2-large-lsp-512-16', 1.538),
    ('LSP-Offload (d=512, r = 8)', 'gpt2-large-lsp-512-8', 1.538),
    ('LSP-Offload (d=512, r = 32)', 'gpt2-large-lsp-512-32', 1.538),
    ('Zero-Offload', 'gpt2-large-zero-offload', 3.14),
    ('LoRA (Rank=16)', 'gpt2-large-lora', 1.17),
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
    df['ppl_mean'] = df['ppl'].rolling(args.rolling, min_periods=1).mean()
    df['ppl_std'] = df['ppl'].rolling(args.rolling, min_periods=1).std() * 0.15

    # Fill the area between the line minus the standard deviation and the line plus the standard deviation
    # ax.fill_between(df['Hours'], df['ppl_mean'] - df['ppl_std'], df['ppl_mean'] + df['ppl_std'], alpha=0.2)

    # Fill the area between the line minus the standard deviation and the line plus the standard deviation
    ax.fill_between(df['Hours'], df['ppl_mean'] - df['ppl_std'], df['ppl_mean'] + df['ppl_std'], alpha=0.2)

    # Plot the rolling mean of 'ppl'
    ax.set_xlabel('Time (h)', fontsize = 30)
    ax.set_ylabel('Perplexity', fontsize = 30)
    # ax.set_xticks(fontsize = 20)
    # ax.set_yticks(fontsize = 20)
    # Plot the rolling mean of 'ppl'
    ax.plot(df['Hours'], df['ppl_mean'], label=name, linewidth = 2.5)
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
print(f'eval result saved to {args.output}_eval.png')
plt.legend(fontsize = 18)
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