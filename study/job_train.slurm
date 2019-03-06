#!/bin/bash
#SBATCH --job-name=rnn_wave
#SBATCH --output="./trained/logs/%A.log"
#SBATCH -p normal
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --mem-per-cpu=20gb

# 120GB for sr=10000, batch_size=9

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$(pwd)

echo "Job started at: $(date +'%d/%m/%y - %H:%m')"
echo ""

srun python -u ./study/train.py --num_threads $SLURM_CPUS_PER_TASK --name $SLURM_JOB_ID \
    --learning_rate 0.001 \
    --N_epochs 20 \
    --gender men \
    --vowels ae ei iy \
    --train_size 90 \
    --test_size 45 \
    --batch_size 9 \
    --c0 1.0 \
    --c1 0.85 \
    --design_region \
    --Nx 200 \
    --Ny 160 \
    --pd 45 \
    --binarize
