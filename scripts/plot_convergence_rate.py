import os
import glob
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def load_data(file: str) -> np.ndarray:
    
    rewards = []

    with open(file, 'r') as f:
        for line in f:
            rewards.append(float(line.split(',')[-1]))
    
    return np.array(rewards)


if __name__ == '__main__':
    
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'figure.figsize': (5.25, 2.5),
        'font.size': 9,
    })
    
    tasks = [
        'StairClimberSparse',
        'MazeSparse',
        'TopOff',
        'FourCorners',
        'Harvester',
        'CleanHouse',
        'DoorKey',
        'OneStroke',
        'Seeder',
        'Snake'
    ]
    
    data_labels = [
        'programmatic',
        'latent',
        # 'latent2',
    ]
    
    data_legends = [
        'Programmatic',
        'Latent',
        # 'Latent, new $N$',
    ]
    
    styles = [
        '-',
        '--',
        '-.'
    ]
    
    n_iterations = 1000
    k=1000
    
    # set top padding to 0.85
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True)
    # fig.supxlabel('Episodic return target')
    fig.supylabel('Convergence rate', fontsize=10, y=0.6)
    
    ax[1,2].set_xlabel('Episodic return target', fontsize=10)
    
    confidence = 0.95
    reward_targets = np.linspace(0, 1, 101)
    for i, task in enumerate(tasks):
        ax[i % 2, i // 2].set_title(f'\sc{{{task.replace("Sparse", "")}}}', fontsize=10)
        ax[i % 2, i // 2].set_ylim(-0.05, 1.05)
        for data_label, legend, style in zip(data_labels, data_legends, styles):
            raw_data = load_data(f'output/rewards_{data_label}_k{k}_{task}.csv')

            convergence_rate = np.zeros((len(reward_targets),))
            convergence_rate_stderr = np.zeros((len(reward_targets),))
                        
            t_value = stats.t.ppf(1 - (1 - confidence) / 2, raw_data.shape[0] - 1)
            
            for j, reward_target in enumerate(reward_targets):
                convergence_rate[j] = (raw_data >= reward_target).mean()
                convergence_rate_stderr[j] = (raw_data >= reward_target).std() / np.sqrt(raw_data.shape[0])
            
            # plt.figure(figsize=(3,3))
            # plt.ylim(-0.05, 1.05)
            # plt.plot(reward_targets, convergence_rate_programmatic, label='Programmatic')
            # plt.fill_between(reward_targets, convergence_rate_programmatic - t_value * convergence_rate_programmatic_stderr, convergence_rate_programmatic + t_value * convergence_rate_programmatic_stderr, alpha=0.2, label='_nolegend_')
            # plt.plot(reward_targets, convergence_rate_latent, label='Latent')
            # plt.fill_between(reward_targets, convergence_rate_latent - t_value * convergence_rate_latent_stderr, convergence_rate_latent + t_value * convergence_rate_latent_stderr, alpha=0.2, label='_nolegend_')
            # plt.legend()
            # plt.title(f'Task: {task}')
            # plt.xlabel('Episodic return target')
            # plt.ylabel('Convergence rate')
            # plt.savefig(f'output/convergence_{task}.png')
            # plt.savefig(f'output/convergence_{task}.pgf')
            # plt.close()
            # ax[i % 2, i // 2].set_xlabel('Episodic return target')
            # ax[i % 2, i // 2].set_ylabel('Convergence rate')
            if i == 0:
                ax[i % 2, i // 2].plot(reward_targets, convergence_rate, style, label=legend, linewidth=1)
            else:
                ax[i % 2, i // 2].plot(reward_targets, convergence_rate, style, label='_nolegend_', linewidth=1)
            ax[i % 2, i // 2].fill_between(reward_targets, convergence_rate - t_value * convergence_rate_stderr, convergence_rate + t_value * convergence_rate_stderr, alpha=0.2, label='_nolegend_')

    
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncols=3, edgecolor='black')
    fig.tight_layout(pad=0.75)
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(f'output/convergence_k{k}.png')
    fig.savefig(f'output/convergence_k{k}.pgf')
    fig.savefig(f'output/convergence_k{k}.pdf')
    
    