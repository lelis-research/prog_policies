import glob
import os
import matplotlib as mpl
from matplotlib import pyplot as plt
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
        data.set_index('num_evaluations', inplace=True)
        
        # include best_reward 0 for the first evaluation
        # data.loc[0] = 0
        
        data_seeds.append(data)
    
    data_seeds = pd.concat(data_seeds, axis=1, sort=True).fillna(method='ffill').fillna(0)
    
    return data_seeds

def plot_results(experiment):
    
    plot_dir = os.path.join('output', experiment, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # check folders in exp_all_tasks/search
    methods = sorted(os.listdir(os.path.join('output', experiment, 'search')))

    all_tasks = []
    for m in methods:
        all_tasks += os.listdir(os.path.join('output', experiment, 'search', m))

    all_tasks = sorted(list(set(all_tasks) - set(['search_args.json'])))

    x_axis = np.logspace(0, 7, 1000)
    
    for task in all_tasks:
        
        median_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
        low_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
        high_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
        min_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
        max_table = pd.DataFrame({'num_evaluations': x_axis}).set_index('num_evaluations')
        
        for method in methods:
            
            directory = os.path.join('output', experiment, 'search', method, task)
            try:
                data = load_data(directory)
            except ValueError:
                continue
            
            median_table[method] = data.median(axis=1).reindex(median_table.index, method='ffill').fillna(0)
            low_table[method] = data.quantile(0.2, axis=1).reindex(low_table.index, method='ffill').fillna(0)
            high_table[method] = data.quantile(0.8, axis=1).reindex(high_table.index, method='ffill').fillna(0)
            min_table[method] = data.min(axis=1).reindex(min_table.index, method='ffill').fillna(0)
            max_table[method] = data.max(axis=1).reindex(max_table.index, method='ffill').fillna(0)
            
            data_last_index = data.index[-1]
            
            # set values in the tables to be NaN after the last index
            median_table.loc[data_last_index:, method] = np.nan
            low_table.loc[data_last_index:, method] = np.nan
            high_table.loc[data_last_index:, method] = np.nan
            min_table.loc[data_last_index:, method] = np.nan
            max_table.loc[data_last_index:, method] = np.nan
            
        plt.figure(figsize=(5, 5))
        plt.suptitle(task)
        plt.xscale('log')
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Best Reward')
        plt.ylim(-0.05, 1.05)
        for method in methods:
            plt.plot(median_table[method], label=method)
            plt.fill_between(median_table.index, low_table[method], high_table[method], alpha=0.2, label='_nolegend_')
        
        if 'DoorKey' in task:
            plt.axhline(0.5, color='black', linestyle='--', label='_nolegend_')
        
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{task}_median_reward.png'))
        plt.close()
        
        plt.figure(figsize=(5, 5))
        plt.suptitle(task)
        plt.xscale('log')
        plt.xlabel('Number of Evaluations')
        plt.ylabel('Best Reward')
        plt.ylim(-0.05, 1.05)
        for method in methods:
            plt.plot(median_table[method], label=method)
            plt.fill_between(median_table.index, min_table[method], max_table[method], alpha=0.2, label='_nolegend_')
        
        if 'DoorKey' in task:
            plt.axhline(0.5, color='black', linestyle='--', label='_nolegend_')
        
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{task}_minmax_reward.png'))
        plt.close()
        
    pass

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('experiment', help='Name of the experiment')
    
    args = parser.parse_args()
    
    plot_results(args.experiment)
