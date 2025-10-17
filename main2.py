import os
import platform

import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
import gymnasium as gym
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils2 import make_env_and_datasets
from data.dataset import Dataset  
from data.replay_buffer import ReplayBuffer 
from data.memory_efficient_replay_buffer import MemoryEfficientReplayBuffer 
from data.data_store import (
    MemoryEfficientReplayBufferDataStore,
)
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb


from flax.core import frozen_dict as flax_froz


FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-double-play-singletask-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 200000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('balanced_sampling', 0, 'Whether to use balanced sampling for online fine-tuning.')

flags.DEFINE_string("aloha_task", "sim_insertion", "aloha task")

config_flags.DEFINE_config_file('agent', 'agents/fql2.py', lock_config=False)


def convert_old_to_new_format(old_dataset_dict):
    """
    Convert from old format (terminals/masks) to new format (dones/masks).
    Old: FrozenDict with 'terminals', 'masks'
    New: dict with 'dones', 'masks'
    
    Args:
        old_dataset_dict: Dict from make_env_and_datasets
    
    Returns:
        new_dataset_dict: Dict compatible with new Dataset class
    """
    new_dict = {}
    for key, value in old_dataset_dict.items():
        if key == 'terminals':
            # Convert terminals to dones (both are needed for compatibility)
            new_dict['dones'] = (value > 0).astype(bool)
            new_dict['masks'] = old_dataset_dict.get('masks', 1.0 - value)
        elif key == 'masks':
            # Already handled with terminals
            if 'masks' not in new_dict:
                new_dict['masks'] = value
        else:
            new_dict[key] = value
    
    # Ensure both dones and masks exist
    if 'dones' not in new_dict and 'terminals' in old_dataset_dict:
        new_dict['dones'] = (old_dataset_dict['terminals'] > 0).astype(bool)
    if 'masks' not in new_dict and 'terminals' in old_dataset_dict:
        new_dict['masks'] = 1.0 - old_dataset_dict['terminals']
    
    return new_dict


def convert_batch_to_agent_format(batch):
    """
    Convert batch from new format to agent-expected format.
    Agent might expect 'terminals' instead of 'dones'.
    """
    batch_dict = dict(batch) if hasattr(batch, 'unfreeze') else batch.copy()
    
    # If agent expects 'terminals', add it
    if 'dones' in batch_dict and 'terminals' not in batch_dict:
        batch_dict['terminals'] = batch_dict['dones'].astype(np.float32)
    
    # Ensure masks exist
    if 'masks' not in batch_dict and 'dones' in batch_dict:
        batch_dict['masks'] = 1.0 - batch_dict['dones'].astype(np.float32)
    
    return batch_dict



def _index_tree(tree, i):
    """tree에서 i번째 샘플만 뽑아 리턴. dict/FrozenDict 내부까지 재귀적으로 들어가 np.ndarray만 슬라이스."""
    if isinstance(tree, (dict, flax_froz.FrozenDict)):
        # FrozenDict면 dict으로 변환 후 처리
        d = dict(tree) if isinstance(tree, flax_froz.FrozenDict) else tree
        return {k: _index_tree(v, i) for k, v in d.items()}
    elif isinstance(tree, np.ndarray):
        return tree[i]
    else:
        # 스칼라/기타 타입은 그대로
        return tree


# def _select_obs_keys(obs, keys):
#     """관측이 Dict일 때 특정 키만 남김. keys=None이면 그대로."""
#     if keys is None or not isinstance(obs, (dict, flax_froz.FrozenDict)):
#         return obs
#     obs = dict(obs) if isinstance(obs, flax_froz.FrozenDict) else obs
#     return {k: obs[k] for k in keys if k in obs}

def populate_replay_buffer_from_dataset(replay_buffer, dataset, chunk=2048):
    """
    dataset: Dataset 클래스(샘플링 메서드 보유) 또는 dict 형태 모두 지원
    obs_key: ('image',) 같이 관측 키 subset 선택. None이면 전체 유지.
    """
    if hasattr(dataset, "__len__") and hasattr(dataset, "dataset_dict"):
        N = len(dataset)
        get_batch = lambda idx: dataset.sample(batch_size=len(idx), indx=np.asarray(idx))
    # elif isinstance(dataset, dict):
    #     # dict가 직접 들어온 경우
    #     def _len_of_dict(d):
    #         for v in d.values():
    #             if isinstance(v, dict):
    #                 return _len_of_dict(v)
    #             if isinstance(v, np.ndarray):
    #                 return len(v)
    #         raise ValueError("cannot infer length from dataset dict")
    #     N = _len_of_dict(dataset)
    #     def get_batch(idx):
    #         # 간단히 전체에서 슬라이스 구현
    #         def _slice_tree(x):
    #             if isinstance(x, np.ndarray):
    #                 return x[idx]
    #             if isinstance(x, (dict, flax_froz.FrozenDict)):
    #                 x = dict(x) if isinstance(x, flax_froz.FrozenDict) else x
    #                 return {k: _slice_tree(v) }  # not used; we will index per item below
    #             return x
    #         return flax_froz.freeze(dataset)
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    added = 0
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        idx = np.arange(start, end, dtype=np.int32)

        # Dataset.sample은 FrozenDict 반환
        batch = get_batch(idx)
        batch = flax_froz.unfreeze(batch) if isinstance(batch, flax_froz.FrozenDict) else batch

        B = end - start
        for t in range(B):
            trans = _index_tree(batch, t) 

            if isinstance(trans.get("actions"), np.ndarray) and trans["actions"].dtype != np.float32:
                trans["actions"] = trans["actions"].astype(np.float32)
            if isinstance(trans.get("rewards"), np.ndarray) and trans["rewards"].dtype != np.float32:
                trans["rewards"] = trans["rewards"].astype(np.float32)
            if isinstance(trans.get("masks"), np.ndarray) and trans["masks"].dtype != np.float32:
                trans["masks"] = trans["masks"].astype(np.float32)
            if isinstance(trans.get("terminals"), np.ndarray) and trans["terminals"].dtype != np.uint8:
                trans["terminals"] = trans["terminals"].astype(np.uint8)

            replay_buffer.insert(trans)
            added += 1

    print(f"[populate_replay_buffer_from_dataset] added {added} transitions.")
    return replay_buffer


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='fql', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    env, eval_env, train_dataset_raw, val_dataset_raw = make_env_and_datasets(
        FLAGS.env_name, 
        frame_stack=FLAGS.frame_stack, 
        aloha_task=FLAGS.aloha_task
    )

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # CHANGED: Convert old format to new format and create Dataset objects
    print("Converting training dataset format...")
    train_dataset_dict = convert_old_to_new_format(train_dataset_raw)
    train_dataset = Dataset(train_dataset_dict, seed=FLAGS.seed)
    print(f"Training dataset size: {len(train_dataset)}")
    
    val_dataset = None
    if val_dataset_raw is not None:
        print("Converting validation dataset format...")
        val_dataset_dict = convert_old_to_new_format(val_dataset_raw)
        val_dataset = Dataset(val_dataset_dict, seed=FLAGS.seed)
        print(f"Validation dataset size: {len(val_dataset)}")

    # CHANGED: Determine if we're using visual environment
    if isinstance(env.observation_space, gym.spaces.Dict):
        obs_keys = list(env.observation_space.spaces.keys())
        is_visual_env = any(k in obs_keys for k in ('image', 'pixels', 'rgb'))
    else:
        is_visual_env = False
    
    pixel_keys = ('image',) if is_visual_env else ()
    reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    def as_dict_space(space: gym.Space, name: str) -> gym.spaces.Dict:
        return space if isinstance(space, gym.spaces.Dict) else gym.spaces.Dict({name: space})

    obs_key = 'observation'
    obs_space_dict = as_dict_space(env.observation_space, obs_key)

    # CHANGED: Create replay buffer based on environment type
    if FLAGS.balanced_sampling:
        # Create separate replay buffer for balanced sampling
        print(f"Creating separate replay buffer (capacity: {FLAGS.buffer_size})...")
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            observation_space=obs_space_dict,
            action_space=env.action_space,
            reward_space=reward_space,
            capacity=FLAGS.buffer_size,
            image_keys=pixel_keys,
        )
        if getattr(replay_buffer, "_num_stack", None) is None:
            replay_buffer._num_stack = 1

    else:
        # Use training dataset as replay buffer
        capacity = max(FLAGS.buffer_size, len(train_dataset) + 1)
        print(f"Creating replay buffer from training data (capacity: {capacity})...")
        
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            observation_space=obs_space_dict,
            action_space=env.action_space,
            reward_space=reward_space,
            capacity=capacity,
            image_keys=pixel_keys,
        )
        if getattr(replay_buffer, "_num_stack", None) is None:
            replay_buffer._num_stack = 1  

        
        # Populate replay buffer with training data
        print(f"Populating replay buffer with {len(train_dataset)} transitions...")
        replay_buffer = populate_replay_buffer_from_dataset(replay_buffer, train_dataset)
        print(f"Replay buffer populated. Size: {len(replay_buffer)}")

    # Create agent.
    print("Creating agent...")
    example_batch = train_dataset.sample(1)
    example_batch = convert_batch_to_agent_format(example_batch)

    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
        print(f"Restoring agent from {FLAGS.restore_path}, epoch {FLAGS.restore_epoch}")
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    
    print(f"Starting training: {FLAGS.offline_steps} offline steps + {FLAGS.online_steps} online steps")
    
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # Offline RL.
            batch = train_dataset.sample(config['batch_size'])
            batch = convert_batch_to_agent_format(batch)

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)
        else:
            # Online fine-tuning.
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()
            
            print("[DEBUG] env.reset ob : ", ob)

            action = agent.sample_actions(observations=ob, temperature=1, seed=key)
            action = np.array(action)

            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            print("[DEBUG] env.step next_ob : ", next_ob)

            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                # Adjust reward for D4RL antmaze.
                reward = reward - 1.0

            # CHANGED: Insert into new replay buffer format
            transition = {
                'observations': {obs_key: ob} if not isinstance(ob, dict) else ob,
                'actions': action,
                'rewards': np.array([reward]) if np.isscalar(reward) else reward,
                'next_observations': {obs_key: next_ob} if not isinstance(next_ob, dict) else next_ob,
                'dones': np.array([done], dtype=np.bool_),
                'masks': np.array([1.0 - float(terminated)], dtype=np.float32),
            }
            replay_buffer.insert(transition)
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent.
            if FLAGS.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer.
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                
                # Merge batches
                batch = {}
                for k in dataset_batch.keys():
                    if k in replay_batch:
                        batch[k] = np.concatenate([dataset_batch[k], replay_batch[k]], axis=0)
                    else:
                        batch[k] = dataset_batch[k]
            else:
                batch = replay_buffer.sample(config['batch_size'])
            
            batch = convert_batch_to_agent_format(batch)

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                val_batch = convert_batch_to_agent_format(val_batch)
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if FLAGS.video_episodes > 0:
                video = get_wandb_video(renders=renders)
                eval_metrics['video'] = video

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)