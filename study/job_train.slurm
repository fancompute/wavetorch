#!/bin/bash
#SBATCH --job-name=rnn_wave
#SBATCH --output="./trained/logs/%A.log"
#SBATCH -p normal
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=20gb

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$(pwd)

echo "Job started at: $(date +'%d/%m/%y - %H:%m')"
echo ""

srun python -u ./study/train.py 						\
					--name $SLURM_JOB_ID				\
					--N_epochs 10 						\
					--train_size 81 					\
					--test_size 30 						\
					--batch_size 9 						\
					--num_threads $SLURM_CPUS_PER_TASK 	\
					--c0 1.0							\
					--c1 0.9 							\
					--binarized
