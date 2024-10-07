#!/bin/bash

## ENVIRONMENT SETTINGS
#SBATCH --get-user-env=L          # Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=download_ImageNet
##SBATCH --job-name=download_food
##SBATCH --job-name=download_cars
#SBATCH --time=72:00:00
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/%x.%j
##SBATCH --gres=gpu:a100:1             # Request 1 GPUs per node
##SBATCH --partition=gpu          # Submit job to the gpu queue

# SBATCH --gres=gpu:a100:1
# SBATCH --gres=gpu:rtx:1 
# SBATCH --gres=gpu:t4:1

# execute python script
#python laion_downloader.py --dataset semi-aves --sampling all
#python laion_downloader.py --dataset flowers102 --sampling random
#python laion_downloader.py --dataset dtd --sampling all
#python laion_downloader.py --dataset eurosat --sampling all
#python laion_downloader.py --dataset fgvc-aircraft --sampling all

#python laion_downloader.py --dataset oxfordpets --sampling random
#python laion_downloader.py --dataset food101 --sampling random
#python laion_downloader.py --dataset stanfordcars --sampling random
python laion_downloader.py --dataset imagenet --sampling random


# Job Environment variables
echo "Job ID: $SLURM_JOBID"
echo "Job submission directory: $SLURM_SUBMIT_DIR"
echo "Temp directory: $TMPDIR"
echo "Scratch directory: $SCRATCH"