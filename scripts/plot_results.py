import glob
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib
from scipy import stats
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def load_data(directory):
    
    seeds = sorted(glob.glob(os.path.join(directory, 'seed_*.csv')))
    
    data_seeds = []
    
    for seed in seeds:

        log = pd.read_csv(seed)
        
        data = pd.DataFrame()
        
        data['num_evaluations'] = log['num_evaluations']
        data['best_reward'] = log['best_reward']

        # Because of checkpointing, we might have duplicates
        data = data.drop_duplicates(subset='num_evaluations', keep='last')
        data.set_index('num_evaluations', inplace=True)
        
        data_seeds.append(data)
    
    data_seeds = pd.concat(data_seeds, axis=1, sort=True).fillna(method='ffill').fillna(0)
    
    return data_seeds


def plot_results(experiment):
    
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'figure.figsize': (5.5, 2.75),
        'font.size': 9,
    })
    
    plot_dir = os.path.join('output', experiment, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # check folders in exp_all_tasks/search
    methods = [
        'StochasticHillClimbing2',
        'LatentCEM_LEAPS',
        'LatentCEM_LEAPS_Original',
    ]
    
    methods_labels = [
        'HC',
        'CEBS',
        'CEM',
    ]
    
    styles = [
        '-',
        '--',
        '-.',
    ]
    
    easy_tasks = [
        'StairClimberSparse_12x12_Crashable',
        'MazeSparse_12x12_Crashable',
        'TopOff_12x12_Crashable',
        'FourCorners_12x12_Crashable',
        'Harvester_8x8_Crashable',
        'CleanHouse_22x14_Crashable',
    ]
    
    easy_tasks_labels = [
        'StairClimber',
        'Maze',
        'TopOff',
        'FourCorners',
        'Harvester',
        'CleanHouse',
    ]
    
    hard_tasks = [
        'DoorKey_8x8_Crashable',
        'OneStroke_8x8_Crashable',
        'Seeder_8x8_Crashable',
        'Snake_8x8_Crashable',
    ]
    
    hard_tasks_labels = [
        'DoorKey',
        'OneStroke',
        'Seeder',
        'Snake',
    ]
    
    all_tasks = easy_tasks + hard_tasks
    all_tasks_labels = easy_tasks_labels + hard_tasks_labels

    x_axis = np.logspace(0, 6, 1000)
    
    all_data = {
        m: pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
        for m in methods
    }
    
    confidence = 0.95
    
    # fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    # fig.supylabel('Best Episodic Return', fontsize=10, y=0.55)
    # fig.supxlabel('Number of Episodes', fontsize=10, y=0.11, x=0.55)
    
    # for i, task in enumerate(easy_tasks):
        
    #     mean_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
    #     high_ci_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
    #     low_ci_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
        
    #     for method in methods:
            
    #         directory = os.path.join('output', experiment, 'search', method, task)
    #         try:
    #             data = load_data(directory)
    #         except ValueError:
    #             continue
            
    #         reindexed_data = data.reindex(x_axis, method='ffill').fillna(0)
            
    #         all_data[method] = pd.concat([all_data[method], reindexed_data], axis=1, sort=True)
            
    #         t_value = stats.t.ppf(1 - (1 - confidence) / 2, len(data) - 1)
            
    #         mean_table[method] = reindexed_data.mean(axis=1)
    #         data_std = reindexed_data.std(axis=1)
    #         high_ci_table[method] = reindexed_data.mean(axis=1) + t_value * data_std / np.sqrt(reindexed_data.shape[1])
    #         low_ci_table[method] = reindexed_data.mean(axis=1) - t_value * data_std / np.sqrt(reindexed_data.shape[1])
            
    #         # data_last_index = data.index[-1]
            
    #         # set values in the tables to be NaN after the last index
    #         # mean_table.loc[data_last_index:, method] = np.nan
    #         # high_ci_table.loc[data_last_index:, method] = np.nan
    #         # low_ci_table.loc[data_last_index:, method] = np.nan
        
    #     ax[i % 2, i // 2].set_title(f'\sc{{{easy_tasks_labels[i]}}}')
    #     ax[i % 2, i // 2].set_ylim(-0.25, 1.05)
    #     for method, label, style in zip(methods, methods_labels, styles):
    #         if i == 0:
    #             ax[i % 2, i // 2].plot(mean_table[method], style, label=label, linewidth=1)
    #         else:
    #             ax[i % 2, i // 2].plot(mean_table[method], style, label='_nolegend_', linewidth=1)
    #         ax[i % 2, i // 2].fill_between(mean_table.index, low_ci_table[method], high_ci_table[method], alpha=0.2, label='_nolegend_')
    #     ax[i % 2, i // 2].set_xscale('log')
        
    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncols=3, edgecolor='black')
    # fig.tight_layout(pad=0.75)
    # fig.subplots_adjust(bottom=0.25)
    # fig.savefig(f'output/easy_tasks.png')
    # fig.savefig(f'output/easy_tasks.pgf')
    
    # fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    # fig.supylabel('Best Episodic Return', fontsize=10, y=0.55)
    # fig.supxlabel('Number of Episodes', fontsize=10, y=0.11, x=0.55)
    
    # for i, task in enumerate(hard_tasks):
        
    #     mean_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
    #     high_ci_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
    #     low_ci_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
        
    #     for method in methods:
            
    #         directory = os.path.join('output', experiment, 'search', method, task)
    #         try:
    #             data = load_data(directory)
    #         except ValueError:
    #             continue
            
    #         reindexed_data = data.reindex(x_axis, method='ffill').fillna(0)
            
    #         all_data[method] = pd.concat([all_data[method], reindexed_data], axis=1, sort=True)
            
    #         t_value = stats.t.ppf(1 - (1 - confidence) / 2, len(data) - 1)
            
    #         mean_table[method] = reindexed_data.mean(axis=1)
    #         data_std = reindexed_data.std(axis=1)
    #         high_ci_table[method] = reindexed_data.mean(axis=1) + t_value * data_std / np.sqrt(reindexed_data.shape[1])
    #         low_ci_table[method] = reindexed_data.mean(axis=1) - t_value * data_std / np.sqrt(reindexed_data.shape[1])
            
    #         # data_last_index = data.index[-1]
            
    #         # set values in the tables to be NaN after the last index
    #         # mean_table.loc[data_last_index:, method] = np.nan
    #         # high_ci_table.loc[data_last_index:, method] = np.nan
    #         # low_ci_table.loc[data_last_index:, method] = np.nan
        
    #     ax[i % 2, i // 2].set_title(f'\sc{{{hard_tasks_labels[i]}}}')
    #     ax[i % 2, i // 2].set_ylim(-0.25, 1.05)
    #     for method, label, style in zip(methods, methods_labels, styles):
    #         if i == 0:
    #             ax[i % 2, i // 2].plot(mean_table[method], style, label=label, linewidth=1)
    #         else:
    #             ax[i % 2, i // 2].plot(mean_table[method], style, label='_nolegend_', linewidth=1)
    #         ax[i % 2, i // 2].fill_between(mean_table.index, low_ci_table[method], high_ci_table[method], alpha=0.2, label='_nolegend_')
    #     ax[i % 2, i // 2].set_xscale('log')
        
    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncols=3, edgecolor='black')
    # fig.tight_layout(pad=0.75)
    # fig.subplots_adjust(bottom=0.25)
    # fig.savefig(f'output/hard_tasks.png')
    # fig.savefig(f'output/hard_tasks.pgf')
    
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True)
    fig.supylabel('Best Episodic Return', fontsize=10, y=0.55)
    fig.supxlabel('Number of Episodes', fontsize=10, y=0.14, x=0.55)
    
    for i, task in enumerate(all_tasks):
        
        mean_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
        high_ci_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
        low_ci_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
        
        for method in methods:
            
            directory = os.path.join('output', experiment, 'search', method, task)
            try:
                data = load_data(directory)
            except ValueError:
                continue
            
            reindexed_data = data.reindex(x_axis, method='ffill').fillna(0)
            
            all_data[method] = pd.concat([all_data[method], reindexed_data], axis=1, sort=True)
            
            t_value = stats.t.ppf(1 - (1 - confidence) / 2, len(data) - 1)
            
            mean_table[method] = reindexed_data.mean(axis=1)
            data_std = reindexed_data.std(axis=1)
            high_ci_table[method] = reindexed_data.mean(axis=1) + t_value * data_std / np.sqrt(reindexed_data.shape[1])
            low_ci_table[method] = reindexed_data.mean(axis=1) - t_value * data_std / np.sqrt(reindexed_data.shape[1])
            
            # data_last_index = data.index[-1]
            
            # set values in the tables to be NaN after the last index
            # mean_table.loc[data_last_index:, method] = np.nan
            # high_ci_table.loc[data_last_index:, method] = np.nan
            # low_ci_table.loc[data_last_index:, method] = np.nan
        
        ax[i % 2, i // 2].set_title(f'\sc{{{all_tasks_labels[i]}}}', fontsize=10)
        ax[i % 2, i // 2].set_ylim(-0.25, 1.05)
        # ax[i % 2, i // 2].set_xlim(1, 1000000)
        for method, label, style in zip(methods, methods_labels, styles):
            if i == 0:
                ax[i % 2, i // 2].plot(mean_table[method], style, label=label, linewidth=1)
            else:
                ax[i % 2, i // 2].plot(mean_table[method], style, label='_nolegend_', linewidth=1)
            ax[i % 2, i // 2].fill_between(mean_table.index, low_ci_table[method], high_ci_table[method], alpha=0.2, label='_nolegend_')
        ax[i % 2, i // 2].set_xscale('log')
        ax[i % 2, i // 2].set_xticks([1, 1000, 1000000])
        ax[i % 2, i // 2].set_yticks([0, 0.5, 1])
        
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncols=3, edgecolor='black')
    fig.tight_layout(pad=0.75)
    fig.subplots_adjust(bottom=0.3)
    fig.savefig(f'output/all_tasks_{experiment}.png')
    fig.savefig(f'output/all_tasks_{experiment}.pgf')
    fig.savefig(f'output/all_tasks_{experiment}.pdf')
        
    # all_mean = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
    # all_low = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
    # all_high = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
    
    # for method in methods:
    #     t_value = stats.t.ppf(1 - (1 - confidence) / 2, len(all_data[method]) - 1)
        
    #     all_mean[method] = all_data[method].mean(axis=1).reindex(all_mean.index, method='ffill').fillna(0)
    #     data_std = all_data[method].std(axis=1).reindex(all_mean.index, method='ffill').fillna(0)
    #     # data_stderr = data_std / np.sqrt(len(all_data[method]))
    #     all_low[method] = all_mean[method] - t_value * data_std / np.sqrt(all_data[method].shape[1])
    #     all_high[method] = all_mean[method] + t_value * data_std / np.sqrt(all_data[method].shape[1])
    #     # all_low[method] = all_data[method].quantile(0.25, axis=1).reindex(all_low.index, method='ffill').fillna(0)
    #     # all_high[method] = all_data[method].quantile(0.75, axis=1).reindex(all_high.index, method='ffill').fillna(0)
    
    # plt.figure(figsize=(5, 5))
    # plt.suptitle('All tasks - aggregate mean reward')
    # plt.xscale('log')
    # plt.xlabel('Number of Evaluations')
    # plt.ylabel('Best Reward')
    # plt.ylim(-0.25, 1.25)
    # for method in methods:
    #     plt.plot(all_mean[method], label=method)
    #     plt.fill_between(all_mean.index, all_low[method], all_high[method], alpha=0.2, label='_nolegend_')
    
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(os.path.join(plot_dir, f'all_tasks_mean_reward.png'))
    # plt.close()
    

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--experiment', help='Name of the experiment', default='exp_iclr_crash')
    
    args = parser.parse_args()
    
    plot_results(args.experiment)
