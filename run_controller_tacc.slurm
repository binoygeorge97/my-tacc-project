#!/bin/bash
#SBATCH -J gru_sweep
#SBATCH -o gru_sweep_%j.out
#SBATCH -e gru_sweep_%j.err
#SBATCH -p gpu-a100                 
#SBATCH -N 3                        
#SBATCH -n 3                        
#SBATCH -t 00:40:00                 
#SBATCH -A IRI26016
#SBATCH --mail-user=phdstudiescontrols@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# 1. Load Core TACC Modules
module purge
module load TACC

# 2. Set your W&B API Key
export WANDB_API_KEY="5b9121798100cc1b68dd94823ced58122a9de2c0"

# 3. Activate Python 3.12 Conda Environment
source $WORK/miniconda3/bin/activate
conda activate nnx_env

# 4. Force JAX to use pip-installed NVIDIA libraries
NV_BASE=$WORK/miniconda3/envs/nnx_env/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV_BASE/cusparse/lib:$NV_BASE/cublas/lib:$NV_BASE/cudnn/lib:$NV_BASE/cuda_runtime/lib:$NV_BASE/cuda_nvrtc/lib:$LD_LIBRARY_PATH

# 5. Identify the Ray Head Node
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

port=6379
ip_head=$head_node_ip:$port
export RAY_ADDRESS=$ip_head

echo "Starting Ray Head on $head_node_ip"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus=128 --num-gpus=3 --block &

sleep 10 

# 6. Start the Ray Worker Nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
if [ "$worker_num" -gt 0 ]; then
    echo "Starting $worker_num Ray Workers..."
    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus=128 --num-gpus=3 --block &
    done
fi

sleep 10 

# 7. Execute your GRU Controller script
echo "Executing GRU Controller Sweep..."
python train_controller_sweep.py

# 8. Cleanup
ray stop
