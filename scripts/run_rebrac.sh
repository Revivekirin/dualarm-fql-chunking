#!/bin/bash
#SBATCH --job-name=sac
#SBATCH --nodelist=pat-t3
#SBATCH --output=log_rl_%j.out
#SBATCH --error=log_rl_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=80G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

conda init
conda activate fql
cd ~/workspace2/fql

export MUJOCO_GL=egl
export EGL_DEVICE_ID=0        # 다중 GPU면 원하는 인덱스
export GYM_DISABLE_PLUGIN_ENTRYPOINTS=1  # 불필요한 플러그인 로딩 방지

python main.py   \
 --env_name=scene-play-singletask-v0   \
 --online_steps=1000000   \
 --agent=agents/rebrac.py   \
 --agent.lr=3e-4   \
 --agent.batch_size=256   \
 --agent.discount=0.99   \
 --agent.tau=0.005   \
 --agent.tanh_squash=True   \
 --agent.actor_fc_scale=0.01   \
 --agent.actor_layer_norm=False   \
 --agent.layer_norm=True   \
 --agent.alpha_actor=0.5   \
 --agent.alpha_critic=0.5   \
 --agent.actor_freq=2   \
 --agent.actor_noise=0.2   \
 --agent.actor_noise_clip=0.5 \
 --video_episodes=5  \
 --save_interval=500000