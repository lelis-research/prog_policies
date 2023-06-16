#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M
#SBATCH --time=0-06:00
#SBATCH --account=def-lelis
#SBATCH --array=1-4

module load python/3.10
module load scipy-stack/2023a
module load cuda/11.0
module load cudnn/8.0.3

source trainer_params/$SLURM_ARRAY_TASK_ID.env
cd ..
source env/bin/activate
python3 scripts/run_trainer.py $params
