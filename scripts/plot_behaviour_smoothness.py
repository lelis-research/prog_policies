import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats    

if __name__ == '__main__':
    
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'figure.figsize': (5.25, 1.5),
        'font.size': 9,
    })
    
    confidence = 0.95
    
    labels = [
        'programmatic',
        'latent_010',
        'latent_025',
        'latent_050',
    ]
    
    legends = [
        'Programmatic',
        'Latent, $\sigma=0.1$',
        'Latent, $\sigma=0.25$',
        'Latent, $\sigma=0.5$',
    ]
    
    styles = [
        '-',
        '--',
        '-.',
        ':'
    ]
    
    means, low_cis, high_cis = [], [], []
    
    for label in labels:
        with open(f'output/{label}_smoothness_values.csv', 'r') as f:
            values = pd.read_csv(f)
        
        t_value = stats.t.ppf(1 - (1 - confidence) / 2, values.shape[0] - 1)
        
        means.append(values.mean(axis=0))
        low_cis.append(values.mean(axis=0) - t_value * values.std(axis=0) / np.sqrt(values.shape[0]))
        high_cis.append(values.mean(axis=0) + t_value * values.std(axis=0) / np.sqrt(values.shape[0]))
    
    sp_means, sp_low_cis, sp_high_cis = [], [], []
    
    for label in labels:
        with open(f'output/{label}_programs.csv', 'r') as f:
            programs = pd.read_csv(f)
        same_program = np.zeros((programs.shape[0],programs.shape[1]-1))
        for i, row in enumerate(programs.values ):
            for j, prog in enumerate(row[1:]):
                same_program[i, j] = (prog == row[0])
        
        t_value = stats.t.ppf(1 - (1 - confidence) / 2, programs.shape[0] - 1)
        
        sp_means.append(same_program.mean(axis=0))
        sp_low_cis.append(same_program.mean(axis=0) - t_value * same_program.std(axis=0) / np.sqrt(same_program.shape[0]))
        sp_high_cis.append(same_program.mean(axis=0) + t_value * same_program.std(axis=0) / np.sqrt(same_program.shape[0]))
    
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=False)
    
    for label, legend, mean, low_ci, high_ci, style in zip(labels, legends, means, low_cis, high_cis, styles):
        ax[0].plot(mean, style, label=legend, linewidth=1)
        ax[0].fill_between(range(len(mean)), low_ci, high_ci, alpha=0.2, label='_nolegend_')
        ax[0].set_ylim(-0.05, 1.05)
        ax[0].set_ylabel('Behavior similarity')
        ax[0].set_xlabel('Number of mutations')
        
    for label, legend, mean, low_ci, high_ci, style in zip(labels, legends, sp_means, sp_low_cis, sp_high_cis, styles):
        ax[1].plot(mean, style, label=legend, linewidth=1)
        ax[1].fill_between(range(len(mean)), low_ci, high_ci, alpha=0.2, label='_nolegend_')
        ax[1].set_ylim(-0.05, 1.05)
        ax[1].set_ylabel('Identity rate')
        ax[1].set_xlabel('Number of mutations')
    
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), edgecolor='black')
    fig.tight_layout(pad=0.75)
    fig.savefig('output/smoothness.png', dpi=600, transparent=True)
    fig.savefig('output/smoothness.pgf')
    fig.savefig('output/smoothness.pdf')
    