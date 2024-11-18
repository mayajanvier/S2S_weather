#!/bin/bash
#
# tf_example_singleGPU.batch
#

#SBATCH --job-name=train_DRUnet+    # create a short name for your job
#SBATCH --cpus-per-gpu=16        # Number of cpus per GPU card (>1 if multi-threaded tasks)
#SBATCH --partition=gpu          # Name of the partition
#SBATCH --gres=gpu:1        # GPU nodes are only available in gpu partition
#SBATCH --mem=128G                # Total memory allocated
#SBATCH --time=1-02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=%x_%j.out   # output file name
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maya.janvier@student-cs.fr  # Replace with your email address   

# Load your environment (conda, ...)
source /home/$USER/.bashrc
conda activate pytorch_cuda
echo "Job started"
date 

#BOTH
# echo "DRUNet prior both +"
# python main.py 

# SINGLE
echo "DRUNet prior temp +"
python main.py -var "2m_temperature" 

echo "Job ended"
date 