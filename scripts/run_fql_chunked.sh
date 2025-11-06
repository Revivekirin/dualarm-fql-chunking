#!/bin/bash
#SBATCH --job-name=fql
#SBATCH --nodelist=pat-t5
#SBATCH --output=log_rl_%j.out
#SBATCH --error=log_rl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=120G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

conda init
conda activate fql
cd ~/workspace2/git/dualarm-fql-chunking

export MUJOCO_GL=egl
export EGL_DEVICE_ID=0        
export GYM_DISABLE_PLUGIN_ENTRYPOINTS=1 

# initial running
python main_chunked.py   \
 --env_name=gym-aloha   \
 --online_steps=3000000  \
 --offline_steps=0 \
 --video_episodes=5  \
 --agent=agents/fql_chunked.py \
 --save_interval=50000 \
 --aloha_task=sim_transfer \
 --restore_path=/home/sophia435256/workspace2/git/dualarm-fql-chunking/exp/fql/Debug/sd000_s_26096.0.20251105_192316 \
 --restore_epoch=400000