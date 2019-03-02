#!/bin/bash
#SBATCH --job-name=rnn_wave
#SBATCH --output="./trained/logs/%A.log"
#SBATCH -p normal
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=8gb

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$(pwd)

echo "Job started at: $(date +'%d/%m/%y - %H:%m')"
echo ""

srun python -u ./study/train.py --num_threads $SLURM_CPUS_PER_TASK --name $SLURM_JOB_ID \
    --N_epochs 10 \
    --train_size 96 \
    --test_size 54 \
    --batch_size 9 \
    --c0 1.0 \
    --c1 0.9 \
    --binarized
