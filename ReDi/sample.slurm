#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-normal
#SBATCH --job-name=sample
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1    
#SBATCH --mem=46G       
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out

cd /vol/csedu-nobackup/project/blont/thesis/
source diff/bin/activate
cd ~/thesis/ReDi

python sample.py SDE --image-size 256 --seed 41 --ckpt /vol/csedu-nobackup/project/blont/thesis/redi/SiT-ReDi-XL-2/model.ckpt