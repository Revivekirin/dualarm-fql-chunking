#!/bin/bash
#SBATCH --job-name=fql
#SBATCH --nodelist=pat-t3
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

python run_teacher_sampling.py   \
 --env_name=scene-play-singletask-v0   \
 --online_steps=1000000  \
 --offline_steps=300000 \
 --video_episodes=5  \
 --agent=agents/fql.py \
 --save_interval=50000 \