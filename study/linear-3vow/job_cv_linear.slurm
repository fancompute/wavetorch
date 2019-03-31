#!/bin/bash
#SBATCH --job-name=cv_linear
#SBATCH --output="./study/vow3/%A.log"
#SBATCH -p normal
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=3500mb

export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export PYTHONPATH=$(pwd)

echo "Job started at: $(date +'%d/%m/%y - %H:%m')"
echo ""

srun python -u -m wavetorch train study/vow3/cv_linear.yml \
    --num_threads $SLURM_CPUS_PER_TASK \
    --name $SLURM_JOB_ID \
    --savedir ./study/vow3/
