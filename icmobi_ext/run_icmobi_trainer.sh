#!/bin/bash
#SBATCH --job-name=ICMOBI_TRAINER # Job name
#SBATCH --mail-type=ALL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=jsalminen@ufl.edu # Where to send mail
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=4 # Number of MPI ranks
#SBATCH --cpus-per-task=4 # Number of tasks on each node
#SBATCH --mem-per-cpu=2000mb # Total memory limit
#SBATCH --distribution=cyclic:cyclic # Distribute tasks cyclically first among nodes and then among sockets within a node
#SBATCH --time=01:00:00 # Time limit hrs:min:sec
#SBATCH --output=/blue/dferris/jsalminen/GitHub/icmobi_extension/_slurm_logs/%j_icmobi_trainer.log # Standard output
#SBATCH --error=/blue/dferris/jsalminen/GitHub/icmobi_extension/_slurm_logs/%j_icmobi_trainer.err # Standard output
#SBATCH --account=dferris # Account name
#SBATCH --qos=dferris-b # Quality of service name
#SBATCH --partition=hpg-default # cluster to run on, use slurm command 'sinfo -s'
#--
# sbatch /blue/dferris/jsalminen/GitHub/icmobi_extension/python_src/icmobi_ext/run_icmobi_trainer.sh
# SBATCH --ntasks-per-node=8
# SBATCH --ntasks-per-socket=4

# lsof -i :29500
# kill -9 <PID>

module purge
# conda activate venv
module load conda gcc/14.2.0 openmpi/5.0.7
conda activate torch_cpu

#--
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
# echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# set linux workspace
if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    TMP_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
    # otherwise: started with bash. Get the real location.
    TMP_PATH=$(realpath $0)
fi

export SCRIPT_DIR=$(dirname $TMP_PATH)
export STUDY_DIR=$SCRIPT_DIR
export SRC_DIR=$(dirname $SCRIPT_DIR)
#-- limiting threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd $STUDY_DIR

# Kick off Python
T1=$(date +%s)
echo "Starting Python script at $(date)"
#-- run script
# srun python -m torch.distributed.run --nproc_per_node=${SLURM_NTASKS} icmobi_trainer_cpu.py
# srun python -m torch.distributed.run --nproc_per_node=$SLURM_NTASKS $SCRIPT_DIR/icmobi_trainer_cpu.py
# srun --mpi=pvmix_v5 python $SCRIPT_DIR/icmobi_trainer_cpu.py # (10/07/2025) JS, I would need to build pytorch from source ot get this working
# srun -n 4 python -m torch.distributed.run --nproc_per_node=4 icmobi_trainer_cpu.py # (10/07/2025) JS, trying now with "gloo" backend
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
NPROC_PER_NODE=$((SLURM_NTASKS / SLURM_JOB_NUM_NODES))
echo "N-Process per Node = $NPROC_PER_NODE"
echo "Master Address = $MASTER_ADDR"


srun python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    icmobi_trainer_cpu.py
# explicitly define gloo for CPU proc (10/07/2025)

#-- log time
T2=$(date +%s)
ELAPSED=$((T2-T1))
echo "Python script ended at $(date) and took $ELAPSED seconds"