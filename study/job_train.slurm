#!/bin/bash
#SBATCH --job-name=rnn_wave
#SBATCH --output="./trained/logs/%A.log"
#SBATCH -p normal
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=20gb

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "$(pwd)"
cd $(pwd)

echo "Job started at: $(date +'%d/%m/%y - %H:%m')"
echo ""

srun /home/iwill/miniconda3/bin/python ./study/train.py --N_epochs 20 --batch_size 9 --num_of_each 36 --num_threads $SLURM_CPUS_PER_TASK --disable_progress
