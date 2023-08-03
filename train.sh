#!/bin/bash
#SBATCH --job-name=pix2pix
#SBATCH --account=a100free
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100-2g-10gb:1
#SBATCH --ntasks=4
#SBATCH --time=48:00:00  # Time limit (HH:MM:SS)
#SBATCH --output=slurm/pix2pix.out  # Output file
#SBATCH --error=slurm/pix2pix.err  # Error file
#SBATCH --mail-type=END,FAIL  # Email you when the job finishes or fails
#SBATCH --mail-user=BRGOLI005@myuct.ac.za # Email address to send to

CUDA_VISIBLE_DEVICES=$(ncvd)

module load python/miniconda3-py310
source activate terrain-a100

# https://stackoverflow.com/questions/75921380/python-segmentation-fault-in-interactive-mode
export LANGUAGE=UTF-8 
export LC_ALL=en_US.UTF-8 
export LANG=UTF-8 
export LC_CTYPE=en_US.UTF-8
export LANG=en_US.UTF-8 
export LC_COLLATE=$LANG 
export LC_CTYPE=$LANG 
export LC_MESSAGES=$LANG 
export LC_MONETARY=$LANG 
export LC_NUMERIC=$LANG 
export LC_TIME=$LANG 
export LC_ALL=$LANG

export WANDB_API_KEY=$(cat wandb_key)

export WANDB__SERVICE_WAIT=300

host=$(hostname)

echo $host

if [[ $host == *"uct"* ]]; then
    output_dir="/scratch/brgoli005/models/planet-pix2pix"
else
    output_dir="/mnt/e/brgoli005/planet-pix2pix"
fi
python -m train \
    --checkpoints_dir $output_dir \
    --batch_size 64 \
    --planet_seed 0 \
    --use_mask_store True \
    --data_dir ./planetAI/data/ \
    --use_wandb \
    --iters 80000 \
    --n_epochs 50 \
    --display_freq 25000 \
    --name planet-pix2pix-0 \
    --size 0 \
    --input_nc 1 \
    --output_nc 1 \
    --wandb_project_name PlanetAI \




conda deactivate
