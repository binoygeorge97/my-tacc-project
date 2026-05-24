#!/bin/bash
#SBATCH -J train-sweep
#SBATCH -o train-sweep.%j.out
#SBATCH -e train-sweep.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:20:00
#SBATCH -p normal

set -euo pipefail

module load python3

python3 -m pip install --user -r requirements.txt
python3 train_sweep.py --output results/sweep_results.csv
