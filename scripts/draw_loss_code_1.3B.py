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
train_losses = pd.read_csv('result_ref/code_train_losses_1.3B.csv')
train_losses.set_index('Step', inplace=True)
# train_losses['Zero Offload'] = train_losses['Full Param']
# train_losses['Native'] = train_losses['Full Param']
print('train_losses.columns', train_losses.columns)
step_times = [
    # ('Lsp-Offload S=1024 d=4', 'Lsp-Offload S=1024 d=4', 115.2),
    ('LSP-Offload (d=1280 r=4)','LSP-Offload d=1280 r=4', 273.),
    ('Lora (Rank=8)', 'Lora rank=8', 170.6),
    # ('Lora rank=128', 'Lora rank=128', 66.56),
    ('GaLore (Rank=256)', 'Galore Value', 170.6),
    ('Zero Offload', 'Full Param', 512.0),
    # ('Native', 'Full Param', 66.56)
]
print('train_losses', train_losses.head(10))
print('step_times', step_times)
for label, name, time_per_iter in step_times:
    tdf = train_losses[[name]]
    # tdf.columns = ['loss']
    # print('tdf', tdf.head(10))
    tdf['Elapsed'] = tdf.index * time_per_iter
    tdf['Hours'] = tdf['Elapsed'] / 3600
    tdf['loss'] = tdf[name].rolling(args.rolling, min_periods = args.min_periods).mean()
    tdf['loss_std'] = tdf[name].rolling(args.rolling, min_periods = args.min_periods).std()
    if name == 'Galore Value':
        name = 'Galore'
    
    tdf['Elapsed'] = tdf.index * time_per_iter
    tdf['Hours'] = tdf['Elapsed'] / 3600
    tdf = tdf[tdf['Hours'] <= args.hours]
    ax.plot(tdf['Hours'], tdf['loss'], label=label, linewidth = 2.5)
    ax.fill_between(tdf['Hours'], tdf['loss'] - tdf['loss_std'], tdf['loss'] + tdf['loss_std'], alpha=0.2)
    ax.set_xlabel('Time (h)', fontsize = 30)
    ax.set_ylabel('Train Loss', fontsize = 30)
    
    print(label, '#sample:', len(tdf['loss']), '#epoch:', len(tdf['loss'])/434 * 5)


    # # Calculate the rolling mean and standard deviation of 'ppl'
    # df['ppl_mean'] = df['ppl'].rolling(args.rolling, min_periods=1).mean()
    # df['ppl_std'] = df['ppl'].rolling(args.rolling, min_periods=1).std() * 0.15

    # # Fill the area between the line minus the standard deviation and the line plus the standard deviation
    # # ax.fill_between(df['Hours'], df['ppl_mean'] - df['ppl_std'], df['ppl_mean'] + df['ppl_std'], alpha=0.2)

    # # Fill the area between the line minus the standard deviation and the line plus the standard deviation
    # ax.fill_between(df['Hours'], df['ppl_mean'] - df['ppl_std'], df['ppl_mean'] + df['ppl_std'], alpha=0.2)

    # # Plot the rolling mean of 'ppl'
    # ax.set_xlabel('Time (h)', fontsize = 30)
    # ax.set_ylabel('Perplexity', fontsize = 30)
    # # ax.set_xticks(fontsize = 20)
    # # ax.set_yticks(fontsize = 20)
    # # Plot the rolling mean of 'ppl'
    # ax.plot(df['Hours'], df['ppl_mean'], label=name, linewidth = 2.5)
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
print(f'eval result saved to {args.output}_train_losses.png')
plt.legend(fontsize = 18)
fig.savefig(f'{args.output}_train_losses.png', bbox_inches = 'tight')
fig.savefig(f'{args.output}_train_losses.pdf', bbox_inches = 'tight')

# fig, ax = plt.subplots()
# for name, path, time_per_iter in paths:
#     df = pd.read_csv(f'models/{path}/losses.csv')
#     df['Elapsed'] = (df['epoch'] * iter_per_epoch + df['iter']) * time_per_iter
#     df['Hours'] = df['Elapsed'] / 3600
#     df = df[df['Hours'] <= args.hours]

#     # Calculate the rolling mean and standard deviation of 'ppl'
#     df['loss_mean'] = df['loss'].rolling(args.rolling, min_periods = args.min_periods).mean()
#     df['loss_std'] = df['loss'].rolling(args.rolling, min_periods = args.min_periods).std()

#     # Fill the area between the line minus the standard deviation and the line plus the standard deviation
#     ax.fill_between(df['Hours'], df['loss_mean'] - df['loss_std'], df['loss_mean'] + df['loss_std'], alpha=0.2)

#     # Plot the rolling mean of 'loss'
#     ax.plot(df['Hours'], df['loss_mean'], label=name)

# ax.legend()
# print(f'training saved to {args.output}_train.png')
# fig.savefig(f'{args.output}_train.png')
        