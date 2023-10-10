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
        'LatentCEM_LEAPS_Original',
        'LatentCEM_LEAPS',
    ]
    
    methods_labels = [
        'Programmatic SHC',
        'Latent CEM (original)',
        'Latent CEM (ours)',
    ]
    
    styles = [
        '-',
        '--',
        '-.',
    ]
    
    easy_tasks = [
        'StairClimberSparse_12x12_LeapsBehaviour',
        'MazeSparse_12x12_LeapsBehaviour',
        'TopOff_12x12_LeapsBehaviour',
        'FourCorners_12x12_LeapsBehaviour',
        'Harvester_8x8_LeapsBehaviour',
        'CleanHouse_22x14_LeapsBehaviour',
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
        'DoorKey_8x8_LeapsBehaviour',
        'OneStroke_8x8_LeapsBehaviour',
        'Seeder_8x8_LeapsBehaviour',
        'Snake_8x8_LeapsBehaviour',
    ]
    
    hard_tasks_labels = [
        'DoorKey',
        'OneStroke',
        'Seeder',
        'Snake',
    ]
    
    all_tasks = easy_tasks + hard_tasks

    x_axis = np.logspace(0, 6, 1000)
    
    for i, task in enumerate(all_tasks):
        
        print(task)
        
        for method in methods:
            
            directory = os.path.join('output', experiment, 'search', method, task)
            try:
                data = load_data(directory)
            except ValueError:
                continue
            
            reindexed_data = data.reindex(x_axis, method='ffill').fillna(0)
            
            final_reward = reindexed_data.iloc[-1, :]
            final_mean = final_reward.mean()
            final_stderr = final_reward.std()
            
            print(f'{method} ${final_mean:.2f}\pm{final_stderr:.2f}$')
        
        print()
        
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
    
    parser.add_argument('--experiment', help='Name of the experiment', default='exp_iclr_lb')
    
    args = parser.parse_args()
    
    plot_results(args.experiment)
