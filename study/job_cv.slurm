#!/bin/bash
#SBATCH --job-name=rnn_wave
#SBATCH --output="./trained/logs/%A.log"
#SBATCH -p normal
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=20gb

# 120GB for sr=10000, batch_size=9

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$(pwd)

echo "Job started at: $(date +'%d/%m/%y - %H:%m')"
echo ""

srun python -u -m wavetorch train \
    --num_threads $SLURM_CPUS_PER_TASK \
    --name $SLURM_JOB_ID \
    --config study/cv_vow3.yml
