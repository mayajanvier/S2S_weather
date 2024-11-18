#!/bin/bash
#
# array.sbatch
#
# Allocated resources are NOT SHARED across the jobs.
# They represent resources allocated for each job
# in the array.

#SBATCH --job-name=inf_EMOS
#SBATCH --output=%x_%A_%a.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G   
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maya.janvier@student-cs.fr  # Replace with your email address                    

# Load your environment (conda, ...)
source /home/$USER/.bashrc
conda activate pytorch_env
echo "Job started"
echo "EMOS inference lead 7,14,21,28,35,39"
date 
python inference.py
echo "Job ended"
date 