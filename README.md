# Programmatic Policies in Karel the Robot

This implements the methods and experiments of [Reclaiming the Source: Programmatic versus Latent Search Spaces](https://openreview.net/forum?id=NGVljI6HkR) published in ICLR 2024.

## Dependencies

We recommend using `conda` to install the dependencies:

```bash
conda create --name prog_policies_env --file environment.yml
```

If `conda` is not available, it is also possible to install dependencies using `pip` on **Python 3.8**:

```bash
pip install -r requirements.txt
```

## Execution

All executable scripts are located in `scripts/`. These are the relevant scripts used in the paper:

- `run_search.py`: Runs a specified search algorithm in a specified task. Includes boilerplate to output log files and checkpoints. Can be plotted with `plot_results.py`.
- `run_search_new.py`: Same as `run_search.py`, but uses a cleaner implementation with no boilerplate.
- `behaviour_smoothness.py`, `same_program_rate.py`, `convergence_rate.py`: Scripts for topology-based evaluation. Plotted with `plot_behaviour_smoothness.py`, `plot_same_program_rate.py`, `plot_convergence_rate.py`, respectively.

All scripts are run with fixed random seeds for reproducibility.

## Parameters for reproducibility

`run_search.py` expects parameters in a JSON format. The parameters used for each search method in each task in the paper are in `sample_args/paper/{HC|CEBS|CEM}/*.json`.

Note that all parameters use `search_seed=1`. To reproduce the results from the paper, run each algorithm with each seed value in `[1,32]`.

