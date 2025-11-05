#!/bin/bash
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-normal
#SBATCH --job-name=eye
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1    
#SBATCH --mem=46G       
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH -e slurms/%j.err        # first create a "slurms" folder in current directory to store logs
#SBATCH -o slurms/%j.out



# Make sure you activate your fmri environment created from src/setup.sh
cd /vol/csedu-nobackup/project/blont/thesis/
source fmri/bin/activate
cd ~/thesis/MindEyeV2/src

# The following line converts your jupyter notebook into a python script runnable with Slurm
jupyter nbconvert Train.ipynb --to python

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=24 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
export MAIN_CACHE_DIR="/vol/csedu-nobackup/project/blont/thesis/mindeyev2/cache"

export HF_HOME="${MAIN_CACHE_DIR}/huggingface"
export TORCH_HOME="${MAIN_CACHE_DIR}/torch"
export WANDB_CACHE_DIR="${MAIN_CACHE_DIR}/wandb"
export CACHE_DIR="${MAIN_CACHE_DIR}" 

mkdir -p $HF_HOME
mkdir -p $TORCH_HOME
mkdir -p $WANDB_CACHE_DIR

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

# multisubject pretraining
model_name="singlesubj_1sess"
echo model_name=${model_name}
accelerate launch \
    --num_processes=$(($NUM_GPUS * $COUNT_NODE)) \
    --num_machines=$COUNT_NODE \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --mixed_precision=fp16 \
    Train.py \
    --data_path=/vol/csedu-nobackup/project/blont/thesis/mindeyev2 \
    --cache_dir=${CACHE_DIR} \
    --model_name=${model_name} \
    --no-multi_subject \
    --subj=1 \
    --batch_size=${BATCH_SIZE} \
    --max_lr=3e-4 \
    --mixup_pct=.33 \
    --num_epochs=150 \
    --use_prior \
    --prior_scale=30 \
    --clip_scale=1 \
    --no-blurry_recon \
    --blur_scale=.5 \
    --no-use_image_aug \
    --n_blocks=4 \
    --hidden_dim=4096 \
    --num_sessions=1 \
    --ckpt_interval=999 \
    --ckpt_saving 
# singlesubject finetuning
#model_name="finetuned_subj01_40sess"
#echo model_name=${model_name}
#accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=/weka/proj-fmri/shared/mindeyev2_dataset --cache_dir=/weka/proj-fmri/shared/cache --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/multisubject_excludingsubj01_40sess
