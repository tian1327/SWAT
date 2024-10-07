#!/bin/bash

## ENVIRONMENT SETTINGS
#SBATCH --get-user-env=L          # Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=sampling_food
#SBATCH --time=1:00:00
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/%x.%j
#SBATCH --gres=gpu:a100:1             # Request 1 GPUs per node
#SBATCH --partition=gpu          # Submit job to the gpu queue

# SBATCH --gres=gpu:a100:1
# SBATCH --gres=gpu:rtx:1 
# SBATCH --gres=gpu:t4:1

# execute python script
#python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset oxfordpets
python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset food101
#python sample_retrieval.py --prefix T2T500+T2I0.25 --num_samples 500 --sampling_method T2T-rank-T2I-tshd --dataset stanfordcars

# Job Environment variables
echo "Job ID: $SLURM_JOBID"
echo "Job submission directory: $SLURM_SUBMIT_DIR"
echo "Temp directory: $TMPDIR"
echo "Scratch directory: $SCRATCH"