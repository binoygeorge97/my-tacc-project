#!/bin/bash
#SBATCH -J nnx_eval
#SBATCH -o nnx_eval_%j.out
#SBATCH -e nnx_eval_%j.err
#SBATCH -p gpu-a100                 # Using 1 GPU node to handle JAX/XLA operations
#SBATCH -N 1                        # Only 1 node needed
#SBATCH -n 1                        # 1 task
#SBATCH -t 00:30:00                 # Short 30 minute runtime limit
#SBATCH -A IRI26016 
#SBATCH --mail-user=phdstudiescontrols@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# 1. Load Core TACC Modules
module purge
module load TACC

# 2. Hardcoded W&B API Key for the evaluation environment
export WANDB_API_KEY="5b9121798100cc1b68dd94823ced58122a9de2c0"

# 3. Activate your custom Python 3.12 Conda Environment
source $WORK/miniconda3/bin/activate
conda activate nnx_env

# 4. Force JAX to use the pip-installed NVIDIA libraries
NV_BASE=$WORK/miniconda3/envs/nnx_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV_BASE/cusparse/lib:$NV_BASE/cublas/lib:$NV_BASE/cudnn/lib:$NV_BASE/cuda_runtime/lib:$NV_BASE/cuda_nvrtc/lib:$LD_LIBRARY_PATH

# 5. Run the evaluation script
echo "Executing Evaluation..."
python evaluate.py
