#!/bin/bash
#SBATCH --job-name=satdamp
#SBATCH --output="./study/satdamp/%A.log"
#SBATCH -p normal
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=6000mb

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$(pwd)

echo "Job started at: $(date +'%d/%m/%y - %H:%m')"
echo ""

srun python -u ./study/vowel_train.py ./study/satdamp/satdamp.yml \
    --num_threads $SLURM_CPUS_PER_TASK \
    --name $SLURM_JOB_ID \
    --savedir ./study/satdamp/
