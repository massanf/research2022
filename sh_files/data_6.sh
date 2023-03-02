#!/bin/sh
#SBATCH -p v
#SBATCH -t 48:0:00
#SBATCH --gres=gpu

export PATH=/home/app/cuda/bin:$PATH
export LD_LIBRARY_PATH=/home/app/cuda/lib64:$LD_LIBRARY_PATH
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# srun /home/u00606/anaconda3/envs/rec/bin/python3.10 testtest.py
# srun python train.py
# srun python3 preparewrap.py
srun /home/u00606/anaconda3/envs/rec/bin/python3.10 preparewrap.py 6