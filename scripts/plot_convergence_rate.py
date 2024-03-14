from __future__ import annotations
import glob
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

def load_data(files: list[str]) -> np.ndarray:
    
    rewards = []

    for file in files:
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
        'font.size': 8,
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
    
    k = 250
    
    fig, ax = plt.subplots(2, 5, sharex=True, sharey=True)
    fig.supxlabel('Episodic return target', fontsize=9, y=0.11)
    fig.supylabel('Convergence rate', fontsize=9, y=0.6)
    
    confidence = 0.95
    reward_targets = np.linspace(0, 1, 101)
    for i, task in enumerate(tasks):
        a = ax[i % 2, i // 2]
        a.set_title(f'\sc{{{task.replace("Sparse", "")}}}', fontsize=9)
        a.set_ylim(-0.05, 1.05)
        a.set_xticks([0.0, 0.5, 1.0])
        a.set_yticks([0.0, 0.5, 1.0])
        
        if task == 'DoorKey':
            axins = a.inset_axes([0.57, 0.65, 0.4, 0.3], xlim=(0.35, 1.01), ylim=(-0.0002, 0.0018))
            axins.set_xticks([0.4, 0.7, 1.0], ["0.4", "0.7", "1"], fontsize=6)
            axins.set_yticks([0.0, 0.001], ["0", "0.001"], fontsize=6)
            for axis in ['top','bottom','left','right']:
                axins.spines[axis].set_linewidth(.5)
            axins.tick_params(width=.5)
            
            a.arrow(0.55, 0.1, 0.05, 0.3, width=0.01, head_width=0.05, head_length=0.05, color='black', length_includes_head=True, alpha=0.8)
            
            # rectpatch, connects = a.indicate_inset_zoom(axins, edgecolor="black")
            # rectpatch.set_visible(False)
            # for c in connects:
            #     c.set_linewidth(.5)
            
        if task == 'Snake':
            axins = a.inset_axes([0.53, 0.65, 0.42, 0.3], xlim=(0.12, 1.01), ylim=(-0.0002, 0.0018))
            axins.set_xticks([0.2, 0.6, 1.0], ["0.2", "0.6", "1"], fontsize=6)
            axins.set_yticks([0.0, 0.001], ["0", "0.001"], fontsize=6)
            for axis in ['top','bottom','left','right']:
                axins.spines[axis].set_linewidth(.5)
            axins.tick_params(width=.5)
            
            a.arrow(0.25, 0.1, 0.15, 0.3, width=0.01, head_width=0.05, head_length=0.05, color='black', length_includes_head=True, alpha=0.8)
            
            # rectpatch, connects = a.indicate_inset_zoom(axins, edgecolor="black")
            # rectpatch.set_visible(False)
            # for c in connects:
            #     c.set_linewidth(.5)
        
        for data_label, legend, style in zip(data_labels, data_legends, styles):
            files = glob.glob(f'output/convergence_results/rewards_{data_label}_k{k}_{task}_seeds*.csv')
            raw_data = load_data(files)

            convergence_rate = np.zeros((len(reward_targets),))
            convergence_rate_stderr = np.zeros((len(reward_targets),))
            convergence_rate_stddev = np.zeros((len(reward_targets),))
                        
            t_value = stats.t.ppf(1 - (1 - confidence) / 2, raw_data.shape[0] - 1)
            
            for j, reward_target in enumerate(reward_targets):
                convergence_rate[j] = (raw_data >= reward_target).mean()
                convergence_rate_stddev[j] = (raw_data >= reward_target).std()
                convergence_rate_stderr[j] = (raw_data >= reward_target).std() / np.sqrt(raw_data.shape[0])

            if i == 0:
                a.plot(reward_targets, convergence_rate, style, label=legend, linewidth=1)
            else:
                a.plot(reward_targets, convergence_rate, style, label='_nolegend_', linewidth=1)
            a.fill_between(reward_targets, convergence_rate - t_value * convergence_rate_stderr, convergence_rate + t_value * convergence_rate_stderr, alpha=0.2, label='_nolegend_')
            # a.fill_between(reward_targets, convergence_rate - convergence_rate_stddev, convergence_rate + convergence_rate_stddev, alpha=0.2, label='_nolegend_')
            
            if task == 'DoorKey' or task == 'Snake':
                axins.plot(reward_targets, convergence_rate, style, label='_nolegend_', linewidth=0.5)
                axins.fill_between(reward_targets, convergence_rate - t_value * convergence_rate_stderr, convergence_rate + t_value * convergence_rate_stderr, alpha=0.2, label='_nolegend_')
    
    fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncols=3, edgecolor='black')
    fig.tight_layout(pad=0.75)
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(f'output/convergence_k{k}.png', dpi=600, transparent=True)
    fig.savefig(f'output/convergence_k{k}.pgf')
    fig.savefig(f'output/convergence_k{k}.pdf')
    
    