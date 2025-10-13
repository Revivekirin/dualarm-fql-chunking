import os
import platform

import json
import random
import time

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb

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

config_flags.DEFINE_config_file('agent', 'agents/fql.py', lock_config=False)

import cv2
import copy

def _resize_top(ob, img_hw):
    if isinstance(ob, dict) and 'pixels' in ob and isinstance(ob['pixels'], dict) and 'top' in ob['pixels']:
        img = ob['pixels']['top']
        if isinstance(img, (np.ndarray,)) and img.ndim == 3:
            h, w = img.shape[:2]
            H, W = img_hw
            if (h, w) != (H, W):
                ob = copy.deepcopy(ob)
                ob['pixels']['top'] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return ob

def _tree_slice_i(tree, i):
    if isinstance(tree, dict):
        return {k: _tree_slice_i(v, i) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        out = [_tree_slice_i(v, i) for v in tree]
        return type(tree)(out)
    try:
        arr = np.asarray(tree)
        if arr.ndim >= 1 and arr.shape[0] > i:
            return arr[i]
        return tree
    except Exception:
        return tree


# def _batch_size_of(batch):
#     leaves = []
#     def _collect(x):
#         leaves.append(x)
#         return x
#     jax.tree_util.tree_map(_collect, batch)
#     for v in leaves:
#         try:
#             a = np.asarray(v)
#             if a.ndim >= 1:
#                 return a.shape[0]
#         except Exception:
#             pass
#     return 1


def _bytes_of_tree(x):
    if isinstance(x, dict):
        return sum(_bytes_of_tree(v) for v in x.values())
    arr = np.asarray(x)
    return arr.nbytes


# -----------------------------------------------------------------------------

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
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, aloha_task=FLAGS.aloha_task
    )

    # --------------------------- 스트리밍 여부 판정 ---------------------------
    is_streaming = hasattr(train_dataset, "stream_batches")

    if not is_streaming and isinstance(train_dataset, dict):
        train_dataset = Dataset.create(**train_dataset)
    if isinstance(val_dataset, dict):
        val_dataset = Dataset.create(**val_dataset)

    # Initialize agent random seeds.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # ----------------------- 스트리밍: 버퍼 생성/워밍업 -----------------------
    stream_iter = None
    replay_buffer = None
    if is_streaming:
        bs = int(config['batch_size'])
        if hasattr(train_dataset, "set_batch_size"):
            train_dataset.set_batch_size(bs)
        stream_iter = train_dataset.stream_batches(batch_size=bs)

        # warm_batch 확보
        try:
            warm_batch = next(stream_iter)
        except StopIteration:
            stream_iter = train_dataset.stream_batches(batch_size=bs)
            warm_batch = next(stream_iter)

        # 메모리 타깃(기본 8GB) 기준으로 버퍼 용량 캡핑
        target_bytes = int(os.getenv("RB_TARGET_BYTES", str(8 * 1024**3)))
        example_transition = {k: _tree_slice_i(v, 0) for k, v in warm_batch.items()}
        bytes_per = _bytes_of_tree(example_transition)
        max_size = max(128, int(target_bytes // max(bytes_per, 1)))
        safe_size = min(FLAGS.buffer_size, max_size)
        if FLAGS.buffer_size > safe_size:
            print(f"[ReplayBuffer] requested_size={FLAGS.buffer_size} -> capped to {safe_size} "
                  f"(~{bytes_per/1e6:.2f} MB/transition, target ~{target_bytes/1e9:.1f} GB)")

        # 버퍼 생성
        replay_buffer = ReplayBuffer.create(example_transition, size=safe_size)

        # warm_batch 일괄 적재
        replay_buffer.add_batch(warm_batch)

        # 추가 워밍업
        target_warm = min(replay_buffer.max_size, 2048)
        while replay_buffer.size < target_warm:
            try:
                stream_batch = next(stream_iter)
            except StopIteration:
                stream_iter = train_dataset.stream_batches(batch_size=bs)
                stream_batch = next(stream_iter)
            replay_buffer.add_batch(stream_batch)

        if FLAGS.balanced_sampling:
            print("[INFO] Streaming mode: ignoring balanced_sampling; sampling only from replay buffer.")

        train_dataset = replay_buffer

    else:
        # ----------------------- 비스트리밍: 기존 로직 -----------------------
        if FLAGS.balanced_sampling:
            example_transition = {k: v[0] for k, v in train_dataset.items()}
            replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
        else:
            train_dataset = ReplayBuffer.create_from_initial_dataset(
                dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
            )
            replay_buffer = train_dataset

    # p_aug / frame_stack 설정
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            if hasattr(dataset, "p_aug"):
                dataset.p_aug = FLAGS.p_aug
            if hasattr(dataset, "frame_stack"):
                dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac' and hasattr(dataset, "return_next_actions"):
                dataset.return_next_actions = True

    # --------------------------- 에이전트 생성용 예시 배치 --------------------------
    if is_streaming:
        # 에이전트 create가 (B=1) 배치를 기대한다면 간단히 [:1]
        example_batch = {k: (v[:1] if isinstance(v, np.ndarray) else v) for k, v in warm_batch.items()}
    else:
        example_batch = train_dataset.sample(1)

    # Create agent.
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent.
    if FLAGS.restore_path is not None:
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

    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # ----------------------- Offline RL -----------------------
            if is_streaming: #TODO: offline RL은 train_dataset에서만
                try:
                    stream_batch = next(stream_iter)
                except StopIteration:
                    stream_iter = train_dataset.stream_batches(batch_size=int(config['batch_size']))
                    stream_batch = next(stream_iter)
                replay_buffer.add_batch(stream_batch)
                batch = replay_buffer.sample(config['batch_size'])
            else:
                batch = train_dataset.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

        else:
            # ----------------------- Online fine-tuning -----------------------
            online_rng, key = jax.random.split(online_rng)

            # if done:
            #     step = 0
            #     ob, _ = env.reset()

            # batched_ob = jax.tree_util.tree_map(lambda x: np.asarray(x)[None, ...], ob)
            # actions_batched = agent.sample_actions(observations=batched_ob, temperature=1, seed=key)
            # action = np.asarray(actions_batched, dtype=np.float32)[0]
            # next_ob, reward, terminated, truncated, info = env.step(action)
            # done = terminated or truncated
            # reset

            if done:
                step = 0
                ob, _ = env.reset()
                ob = _resize_top(ob, config.get('img_hw', (240, 320)))

            # 액션 샘플링
            batched_ob = jax.tree_util.tree_map(lambda x: np.asarray(x)[None, ...], ob)
            actions_batched = agent.sample_actions(observations=batched_ob, temperature=1, seed=key)
            action = np.asarray(actions_batched, dtype=np.float32)[0]

            # step
            next_ob, reward, terminated, truncated, info = env.step(action)
            next_ob = _resize_top(next_ob, config.get('img_hw', (240, 320)))  
            done = terminated or truncated



            if 'antmaze' in FLAGS.env_name and (
                'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
            ):
                reward = reward - 1.0

            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            if (FLAGS.balanced_sampling and not is_streaming):
                dataset_batch = train_dataset.sample(config['batch_size'] // 2)
                replay_batch = replay_buffer.sample(config['batch_size'] // 2)
                batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
            else:
                batch = replay_buffer.sample(config['batch_size'])

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
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
