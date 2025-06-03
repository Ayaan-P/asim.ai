#!/bin/bash
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1  
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4 
#SBATCH --mem=32G         
#SBATCH --output=cluster_generate_output_%j.log # Output log file, %j is job ID

module purge
module load python/3.10.1 
source /software/anaconda3/2018.12/etc/profile.d/conda.sh 
conda activate asimai-env 


python cluster_generate.py \
    --max_new_tokens 100 \
    --num_return_sequences 3 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.95


echo "Generation script finished."