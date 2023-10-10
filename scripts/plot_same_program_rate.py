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
        'figure.figsize': (4.5, 2.5)
    })
    
    confidence = 0.95
    
    labels = [
        'programmatic',
        'latent_010',
        'latent_025',
        'latent_050',
        # 'latent2_010',
        # 'latent2_025',
        # 'latent2_050',
    ]
    
    legends = [
        'Programmatic',
        'Latent, $\sigma=0.1$',
        'Latent, $\sigma=0.25$',
        'Latent, $\sigma=0.5$',
        # 'Latent new $N$, $\sigma=0.1$',
        # 'Latent new $N$, $\sigma=0.25$',
        # 'Latent new $N$, $\sigma=0.5$',
    ]
    
    means, low_cis, high_cis = [], [], []
    
    for label in labels:
        with open(f'output/{label}_programs.csv', 'r') as f:
            programs = pd.read_csv(f)
        same_program = np.zeros((programs.shape[0],programs.shape[1]-1))
        for i, row in enumerate(programs.values ):
            for j, prog in enumerate(row[1:]):
                same_program[i, j] = (prog == row[0])
        
        t_value = stats.t.ppf(1 - (1 - confidence) / 2, programs.shape[0] - 1)
        
        means.append(same_program.mean(axis=0))
        low_cis.append(same_program.mean(axis=0) - t_value * same_program.std(axis=0) / np.sqrt(same_program.shape[0]))
        high_cis.append(same_program.mean(axis=0) + t_value * same_program.std(axis=0) / np.sqrt(same_program.shape[0]))
        
    plt.figure()
    for legend, mean, low_ci, high_ci in zip(legends, means, low_cis, high_cis):
        plt.plot(mean, label=legend)
        plt.fill_between(range(len(mean)), low_ci, high_ci, alpha=0.2, label='_nolegend_')
    # plt.text(1.5, 0.39, f'Programmatic', color='C0')
    # plt.text(6, 0.24, f'Latent, $\sigma=0.25$', color='C1')
    # plt.text(6, 0.45, f'Latent, $\sigma=0.1$', color='C2')
    # plt.text(5, 0.09, f'Latent, $\sigma=0.5$', color='C3')
    plt.legend()
    plt.xlabel('Number of mutations')
    plt.ylabel('``Same program\'\' rate')
    plt.tight_layout()
    plt.savefig('output/same_program.png')
    plt.savefig('output/same_program.pgf')
    plt.close()
    