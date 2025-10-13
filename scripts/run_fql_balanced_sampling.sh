#!/bin/bash
#SBATCH --job-name=fql
#SBATCH --nodelist=pat-t5
#SBATCH --output=log_rl_%j.out
#SBATCH --error=log_rl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

conda init
conda activate fql
cd ~/workspace2/dualarm-fql-chunking

export MUJOCO_GL=egl
export EGL_DEVICE_ID=0        
export GYM_DISABLE_PLUGIN_ENTRYPOINTS=1 


python main.py   \
 --env_name=gym-aloha   \
 --online_steps=1000000  \
 --offline_steps=0 \
 --video_episodes=5  \
 --agent=agents/fql.py \
 --save_interval=100000 \
 --aloha_task=sim_insertion \
 --balanced_sampling=1 \
 --restore_path=/home/sophia435256/workspace2/dualarm-fql-chunking/exp/fql/Debug/sd000_s_22778.0.20251004_165554 \
 --restore_epoch=1000000 \ 

 # initial running
# python main.py   \
#  --env_name=gym-aloha   \
#  --online_steps=1000000  \
#  --offline_steps=300000 \
#  --video_episodes=5  \
#  --agent=agents/fql.py \
#  --save_interval=100000 \
#  --aloha_task=sim_insertion \
#  --balanced_sampling=1