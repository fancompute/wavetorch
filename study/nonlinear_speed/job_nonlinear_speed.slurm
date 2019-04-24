#!/bin/bash
#SBATCH --job-name=nonlinear_speed
#SBATCH --output="./study/kerr-3vow/%A.log"
#SBATCH -p normal
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4500mb

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$(pwd)

echo "Job started at: $(date +'%d/%m/%y - %H:%m')"
echo ""

srun python -u ./study/vowel_train.py ./study/nonlinear_speed/nonlinear_speed.yml \
    --num_threads $SLURM_CPUS_PER_TASK \
    --name $SLURM_JOB_ID \
    --savedir ./study/nonlinear_speed/
