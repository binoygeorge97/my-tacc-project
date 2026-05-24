#!/bin/bash
#SBATCH -J nnx_sweep
#SBATCH -o nnx_sweep_%j.out
#SBATCH -e nnx_sweep_%j.err
#SBATCH -p gpu-a100                 # Change to gpu-h100 if you want the H100 nodes
#SBATCH -N 3                        # Requesting 3 nodes
#SBATCH -n 3                        # 3 tasks (1 per node)
#SBATCH -t 01:00:00                 # Max runtime (1 hour)
#SBATCH -A IRI26016                 # <--- REPLACE WITH YOUR TACC PROJECT CODE

# 1. Load Modules
module load python3/3.9.2
module load cuda/12.0

# Add your W&B API Key here!
export WANDB_API_KEY="5b9121798100cc1b68dd94823ced58122a9de2c0"

# 2. Setup Virtual Environment (Runs once if it doesn't exist)
if [ ! -d "$WORK/nnx_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $WORK/nnx_env
    source $WORK/nnx_env/bin/activate
    pip install -U pip
    pip install -r requirements.txt
else
    source $WORK/nnx_env/bin/activate
fi

# 3. Identify the Ray Head Node
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

sleep 10 # Wait for head node

# 4. Start the Ray Worker Nodes
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

sleep 10 # Wait for workers

# 5. Execute your Python script
echo "Executing Python Sweep..."
python train_sweep.py

# 6. Cleanup
ray stop
