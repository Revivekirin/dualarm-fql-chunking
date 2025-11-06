#!/bin/bash
#SBATCH --job-name=fql
#SBATCH --nodelist=pat-t6
#SBATCH --output=log_rl_%j.out
#SBATCH --error=log_rl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

conda init
conda activate fql
cd ~/workspace2/git/dualarm-fql-chunking

export MUJOCO_GL=egl
export EGL_DEVICE_ID=0        
export GYM_DISABLE_PLUGIN_ENTRYPOINTS=1 

# python main2.py   \
#  --env_name=gym-aloha   \
#  --online_steps=1000000  \
#  --offline_steps=0 \
#  --agent=agents/fql2.py \



# initial running
# python run_teacher_sampling.py   \
# python main.py    \
#  --env_name=gym-aloha   \
#  --online_steps=1000000  \
#  --offline_steps=300000 \
#  --video_episodes=5  \
#  --agent=agents/fql.py \
#  --save_interval=50000 \
#  --aloha_task=sim_insertion \
#  --restore_path=/home/sophia435256/workspace2/git/dualarm-fql-chunking/exp/fql/Debug/sd000_s_23779.0.20251015_133357 \
#  --restore_epoch=300000 \ 
#  --balanced_sampling=1 \


 # checkpoint load
python main.py    \
 --env_name=gym-aloha   \
 --online_steps=3000000   \
 --offline_steps=0  \
 --video_episodes=5   \
 --agent=agents/fql.py  \
 --save_interval=500000 \
 --restore_path=/home/sophia435256/workspace2/git/dualarm-fql-chunking/exp/fql/Debug/sd000_s_26072.0.20251105_173025 \
 --restore_epoch=1300000