#!/bin/bash

## ENVIRONMENT SETTINGS
#SBATCH --get-user-env=L          # Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=feature_extraction_cars
#SBATCH --time=20:00:00
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
#python extract_mined_feature.py --dataset oxfordpets --model_cfg vitb32_openclip_laion400m
#python extract_mined_feature.py --dataset food101 --model_cfg vitb32_openclip_laion400m
python extract_mined_feature.py --dataset stanfordcars --model_cfg vitb32_openclip_laion400m
#python extract_mined_feature.py --dataset imagenet --model_cfg vitb32_openclip_laion400m

# Job Environment variables
echo "Job ID: $SLURM_JOBID"
echo "Job submission directory: $SLURM_SUBMIT_DIR"
echo "Temp directory: $TMPDIR"
echo "Scratch directory: $SCRATCH"