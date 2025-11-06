# main.py
import os
import platform

import json
import random
import time
import csv

import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate_chunked, flatten
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

config_flags.DEFINE_config_file('agent', 'agents/fql_chunked.py', lock_config=False)

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

def _bytes_of_tree(x):
    if isinstance(x, dict):
        return sum(_bytes_of_tree(v) for v in x.values())
    arr = np.asarray(x)
    return arr.nbytes

# ----------------------------
# 로깅/스냅샷 유틸
# ----------------------------
def _append_row_to_csv(path, row: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)

def _pca2(x_np: np.ndarray):
    x = x_np - x_np.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(x, full_matrices=False)
    P = VT[:2].T
    return x @ P, P

def _export_bcflow_vector_field(agent, ob_ref, out_path,
                                grid_min=-2.0, grid_max=2.0, grid_n=21,
                                t_list=(0.25, 0.5, 0.75)):
    xs = np.linspace(grid_min, grid_max, grid_n, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    G = grid.shape[0]

    D = int(agent.config["action_dim"])
    if D == 2:
        X = grid
        P = np.eye(2, dtype=np.float32)
    else:
        rng = np.random.default_rng(0)
        P = rng.normal(size=(D, 2)).astype(np.float32)
        P /= (np.linalg.norm(P, axis=0, keepdims=True) + 1e-8)
        X = grid @ P.T

    ob_batched = jax.tree_util.tree_map(lambda x: np.asarray(x)[:1], ob_ref)
    ob_batched = jax.tree_util.tree_map(
        lambda x: np.repeat(x, G, axis=0) if hasattr(x, "shape") and x.ndim >= 1 else x,
        ob_batched
    )

    out = []
    for t_val in t_list:
        t = np.full((G, 1), t_val, dtype=np.float32)
        vels = agent.network.select("actor_bc_flow")(ob_batched, X, t, is_encoded=False)
        vels = np.asarray(vels)
        V2 = vels @ P if D > 2 else vels
        out.append({
            "t": float(t_val),
            "points": grid.tolist(),
            "vectors": V2.tolist(),
        })
    _save_json(out_path, {"vector_field": out})

def _export_student_vs_teacher(agent, ob_ref, out_path, N=256):
    D = int(agent.config["action_dim"])
    key = jax.random.PRNGKey(0)
    noises = jax.random.normal(key, (N, D))

    ob_batched = jax.tree_util.tree_map(lambda x: np.asarray(x)[:1], ob_ref)
    ob_batched = jax.tree_util.tree_map(
        lambda x: np.repeat(x, N, axis=0) if hasattr(x, "shape") and x.ndim >= 1 else x,
        ob_batched
    )

    acts_student = agent.network.select("actor_onestep_flow")(ob_batched, noises)
    acts_student = np.asarray(acts_student)

    acts_teacher = agent.compute_flow_actions(ob_batched, noises)
    acts_teacher = np.asarray(acts_teacher)

    X = np.concatenate([acts_teacher, acts_student], axis=0)
    X2, _ = _pca2(X)
    T2 = X2[:N]
    S2 = X2[N:]
    _save_json(out_path, {"teacher": T2.tolist(), "student": S2.tolist()})

# ----------------------------
# 청크 배치 변환 유틸
# ----------------------------
def _make_chunked_batch(batch, H, gamma):
    """
    입력 batch(스텝 단위)를 길이 H의 청크 배치로 변환.
    필요한 키: observations, actions, rewards, masks, next_observations, terminals
    반환 키:
      observations          : (T, ...)
      actions_chunk         : (T, H, d_a)
      rewards_h             : (T,)
      masks_h               : (T,)
      next_observations_h   : (T, ...)
    """
    obs = np.asarray(batch['observations'])
    act = np.asarray(batch['actions'])
    rew = np.asarray(batch['rewards']).reshape(-1)
    msk = np.asarray(batch['masks']).reshape(-1)
    nxt = np.asarray(batch['next_observations'])
    ter = np.asarray(batch['terminals']).reshape(-1)

    B = act.shape[0]
    T = max(0, B - H)
    if T == 0:
        raise ValueError(f"Batch too short for chunking: B={B} < H={H}")

    obs_t = obs[:T]
    a_chunks = np.stack([act[i:i+H] for i in range(T)], axis=0)  # (T,H,d_a)

    gammas = (gamma ** np.arange(H)).astype(rew.dtype)
    rew_mat = np.stack([rew[i:i+H] for i in range(T)], axis=0)   # (T,H)
    r_h = (rew_mat * gammas[None, :]).sum(axis=1)                # (T,)

    term_mat = np.stack([ter[i:i+H] for i in range(T)], axis=0)  # (T,H)
    ended_inside = term_mat.max(axis=1) > 0
    masks_h = (~ended_inside).astype(rew.dtype)                  # (T,)

    next_obs_h = nxt[H-1: H-1+T]
    return dict(
        observations=obs_t,
        actions_chunk=a_chunks,
        rewards_h=r_h,
        masks_h=masks_h,
        next_observations_h=next_obs_h,
    )

# -----------------------------------------------------------------------------


def main(_):
    # logger setup
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='fql', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # portfolio outputs
    PORT_DIR = os.path.join(FLAGS.save_dir, "portfolio_logs")
    os.makedirs(PORT_DIR, exist_ok=True)
    LC_PATH  = os.path.join(PORT_DIR, "learning_curves.csv")
    VF_PATH  = os.path.join(PORT_DIR, "vector_field_bcflow.json")
    EMB_PATH = os.path.join(PORT_DIR, "embedding_student_teacher.json")

    # env & datasets
    config = FLAGS.agent
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name, frame_stack=FLAGS.frame_stack, aloha_task=FLAGS.aloha_task
    )

    # chunk config
    chunk_size = int(config['chunk_len'])           
    gamma = float(config['discount'])

    # streaming?
    is_streaming = hasattr(train_dataset, "stream_batches")

    if not is_streaming and isinstance(train_dataset, dict):
        train_dataset = Dataset.create(**train_dataset)
    if isinstance(val_dataset, dict):
        val_dataset = Dataset.create(**val_dataset)

    # seeds
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # streaming: warm up replay
    stream_iter = None
    replay_buffer = None
    if is_streaming:
        bs = int(config['batch_size'])
        if hasattr(train_dataset, "set_batch_size"):
            train_dataset.set_batch_size(bs)
        stream_iter = train_dataset.stream_batches(batch_size=bs)

        try:
            warm_batch = next(stream_iter)
        except StopIteration:
            stream_iter = train_dataset.stream_batches(batch_size=bs)
            warm_batch = next(stream_iter)

        target_bytes = int(os.getenv("RB_TARGET_BYTES", str(8 * 1024**3)))
        example_transition = {k: _tree_slice_i(v, 0) for k, v in warm_batch.items()}
        bytes_per = _bytes_of_tree(example_transition)
        max_size = max(128, int(target_bytes // max(bytes_per, 1)))
        safe_size = min(FLAGS.buffer_size, max_size)
        if FLAGS.buffer_size > safe_size:
            print(f"[ReplayBuffer] requested_size={FLAGS.buffer_size} -> capped to {safe_size} "
                  f"(~{bytes_per/1e6:.2f} MB/transition, target ~{target_bytes/1e9:.1f} GB)")

        replay_buffer = ReplayBuffer.create(example_transition, size=safe_size)
        replay_buffer.add_batch(warm_batch)

        target_warm = min(replay_buffer.max_size, 1024)
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
        if FLAGS.balanced_sampling:
            example_transition = {k: v[0] for k, v in train_dataset.items()}
            replay_buffer = ReplayBuffer.create(example_transition, size=FLAGS.buffer_size)
        else:
            train_dataset = ReplayBuffer.create_from_initial_dataset(
                dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
            )
            replay_buffer = train_dataset

    # dataset aug flags
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            if hasattr(dataset, "p_aug"):
                dataset.p_aug = FLAGS.p_aug
            if hasattr(dataset, "frame_stack"):
                dataset.frame_stack = FLAGS.frame_stack
            if config['agent_name'] == 'rebrac' and hasattr(dataset, "return_next_actions"):
                dataset.return_next_actions = True

    # -------- example batch (chunked) for agent.create --------
    def _safe_example_chunk(dset, H, gamma):
        """샘플이 짧아 실패하면 복제해서 길이를 H+1로 맞춰 만든다."""
        try:
            raw = dset.sample(max(H + 1, 2))
            return _make_chunked_batch(raw, H=H, gamma=gamma)
        except Exception:
            raw = dset.sample(2)  # 최소 2
            # 첫 행 복제해서 길이 H+1 만들기
            for k in raw:
                arr = np.asarray(raw[k])
                reps = H + 1 - arr.shape[0]
                if reps > 0:
                    raw[k] = np.concatenate([arr, np.repeat(arr[:1], reps, axis=0)], axis=0)
            return _make_chunked_batch(raw, H=H, gamma=gamma)

    example_chunk = _safe_example_chunk(replay_buffer, chunk_size, gamma)
    print("[MAIN] example_chunk sizes:",
          "obs", example_chunk['observations'].shape,
          "act_chunk", example_chunk['actions_chunk'].shape)

    # create agent (ex_actions는 (1,H,d_a)여야 함)
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_chunk['observations'][:1],
        example_chunk['actions_chunk'][:1],
        config,
    )

    # restore
    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    # train
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger  = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time  = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    last_success_rate = 0.0

    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + FLAGS.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= FLAGS.offline_steps:
            # ------- Offline RL: 샘플 → 청크화 → 업데이트 -------
            raw_batch = train_dataset.sample(config['batch_size'])
            batch = _make_chunked_batch(raw_batch, H=chunk_size, gamma=gamma)

            if config['agent_name'] == 'rebrac':
                agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
            else:
                agent, update_info = agent.update(batch)

        else:
            # ------- Online fine-tuning -------
            online_rng, key = jax.random.split(online_rng)

            if done:
                step = 0
                ob, _ = env.reset()
                ob = _resize_top(ob, config.get('img_hw', (240, 320)))

            batched_ob = jax.tree_util.tree_map(lambda x: np.asarray(x)[None, ...], ob)

            # 정책 출력: (1, AH) -> (AH,)
            actions_batched = agent.sample_actions(observations=batched_ob, temperature=1, seed=key)
            a_flat = np.asarray(actions_batched, dtype=np.float32)[0]

            # 권장: 단일 액션 차원은 env에서 읽어오면 안전
            Da = int(getattr(config, 'action_dim_single', getattr(env.action_space, 'shape', (None,))[0]))
            if Da is None:
                raise ValueError("Cannot infer action_dim_single; set config['action_dim_single'] or use env.action_space.shape[0].")
            a_chunk = a_flat.reshape(chunk_size, Da)   # (H, Da)

            total_reward = 0.0
            terminated = truncated = False
            info = {}
            next_ob = ob

            # ── 오픈루프 chunk_size-스텝 실행 ──
            for k in range(chunk_size):
                action_k = a_chunk[k].astype(np.float32)      # (Da,)
                try:
                    next_ob, reward, terminated, truncated, info = env.step(action_k)
                    next_ob = _resize_top(next_ob, config.get('img_hw', (240, 320)))
                except Exception as e:
                    print(f"[ONLINE] physics error at inner step {k}: {e}")
                    # 비정상 상황: 보수적으로 에피소드 종료 처리
                    reward, terminated, truncated = 0.0, True, False
                    info = {}

                # 리플레이에 매 스텝 적재(원하면 첫/마지막만 적재하도록 바꿔도 됨)
                replay_buffer.add_transition(dict(
                    observations=ob,
                    actions=action_k,
                    rewards=reward,
                    terminals=float(terminated or truncated),
                    masks=1.0 - float(terminated),
                    next_observations=next_ob,
                ))

                total_reward += float(reward)
                ob = next_ob
                step += 1

                if terminated or truncated:
                    break

            done = terminated or truncated

            if 'antmaze' in FLAGS.env_name and ('diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name):
                # 필요하면 per-step이 아니라 total_reward에 적용하도록 조정
                total_reward -= 1.0

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

        if (FLAGS.balanced_sampling and not is_streaming):
            dataset_batch_raw = train_dataset.sample(config['batch_size'] // 2)
            replay_batch_raw  = replay_buffer.sample(config['batch_size'] // 2)
            dataset_batch = _make_chunked_batch(dataset_batch_raw, H=chunk_size, gamma=gamma)
            replay_batch  = _make_chunked_batch(replay_batch_raw,  H=chunk_size, gamma=gamma)
            batch = {k: np.concatenate([dataset_batch[k], replay_batch[k]], axis=0) for k in dataset_batch}
        else:
            raw_batch = replay_buffer.sample(config['batch_size'])
            batch = _make_chunked_batch(raw_batch, H=chunk_size, gamma=gamma)

        if config['agent_name'] == 'rebrac':
            agent, update_info = agent.update(batch, full_update=(i % config['actor_freq'] == 0))
        else:
            agent, update_info = agent.update(batch)

        # ------- Log -------
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}

            if val_dataset is not None:
                # 검증도 청크 형태로
                try:
                    val_raw = val_dataset.sample(config['batch_size'])
                except Exception:
                    # val_dataset이 작으면 train 쪽에서 대체
                    val_raw = train_dataset.sample(config['batch_size'])
                val_batch = _make_chunked_batch(val_raw, H=chunk_size, gamma=gamma)
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})

            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

            _append_row_to_csv(LC_PATH, {
                "step": int(i),
                "reward": float(update_info.get("actor/q", 0.0)),
                "success_rate": float(last_success_rate),
                "distil_loss": float(update_info.get("distill_loss", update_info.get("actor/distill_loss", 0.0))),
                "bc_flow_loss": float(update_info.get("bc_flow_loss", update_info.get("actor/bc_flow_loss", 0.0))),
                "q_loss": float(update_info.get("q_loss", update_info.get("actor/q_loss", 0.0))),
                "critic_loss": float(update_info.get("critic_loss", update_info.get("critic/critic_loss", 0.0))),
                "mse": float(update_info.get("mse", update_info.get("actor/mse", 0.0))),
            })

        # ------- Evaluate -------
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
            renders = []
            eval_metrics = {}
            eval_info, trajs, cur_renders = evaluate_chunked(
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

            if "success_rate" in eval_info:
                last_success_rate = float(eval_info["success_rate"])
            elif "avg_success_rate" in eval_info:
                last_success_rate = float(eval_info["avg_success_rate"])

            # 스냅샷(BC-flow 벡터필드 / 분포)
            try:
                if val_dataset is not None:
                    ref_raw = val_dataset.sample(max(chunk_size + 1, 2))
                else:
                    ref_raw = train_dataset.sample(max(chunk_size + 1, 2))
                # 시각화용은 관측만 필요 — 첫 시점 관측으로 충분
                ob_ref = {"observations": np.asarray(ref_raw["observations"][:1])}
                _export_bcflow_vector_field(agent, ob_ref, out_path=VF_PATH)
                _export_student_vs_teacher(agent, ob_ref, out_path=EMB_PATH)
            except Exception as e:
                print(f"[WARN] snapshot export failed at step {i}: {e}")

            _append_row_to_csv(LC_PATH, {
                "step": int(i),
                "reward": float(update_info.get("actor/q", 0.0)),
                "success_rate": float(last_success_rate),
                "distil_loss": float(update_info.get("distill_loss", update_info.get("actor/distill_loss", 0.0))),
                "bc_flow_loss": float(update_info.get("bc_flow_loss", update_info.get("actor/bc_flow_loss", 0.0))),
                "q_loss": float(update_info.get("q_loss", update_info.get("actor/q_loss", 0.0))),
                "critic_loss": float(update_info.get("critic_loss", update_info.get("critic/critic_loss", 0.0))),
                "mse": float(update_info.get("mse", update_info.get("actor/mse", 0.0))),
            })

        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
