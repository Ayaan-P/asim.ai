#!/bin/bash
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --output=%x-%j.out

# Load conda
module purge
module load python/3.10.1
source /software/anaconda3/2018.12/etc/profile.d/conda.sh
conda activate asimai-env

# Optional: set HuggingFace token
export HF_TOKEN="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"

# Launch
accelerate launch --num_processes=2 --mixed_precision bf16 train_asim_lora.py
