#!/bin/sh
#SBATCH -p v
#SBATCH -t 24:0:00
#SBATCH --gres=gpu:8
export PATH=/home/app/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/app/cuda/lib64:$LD_LIBRARY_PATH

echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# needs `conda activate DiffDRR`

srun python digitally_reconstructed_radiograph/create_drr\ copy.py --rotate_num 1 --vol 1 --num_views 400