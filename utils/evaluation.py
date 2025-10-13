from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
import copy
import cv2

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def _standardize_observation(ob, img_hw):
    """Return a copy of `ob` with ob['pixels']['top'] resized to `img_hw`=(H,W) if present."""
    if not isinstance(ob, dict):
        return ob
    if 'pixels' in ob and isinstance(ob['pixels'], dict) and 'top' in ob['pixels']:
        img = ob['pixels']['top']
        # numpy array RGB(H,W,3)만 처리
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[-1] == 3:
            H, W = img_hw
            h, w = img.shape[:2]
            if (h, w) != (H, W):
                ob = copy.deepcopy(ob)
                ob['pixels']['top'] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return ob



# def evaluate(
#     agent,
#     env,
#     config=None,
#     num_eval_episodes=50,
#     num_video_episodes=0,
#     video_frame_skip=3,
#     eval_temperature=0,
# ):
#     """Evaluate the agent in the environment.

#     Args:
#         agent: Agent.
#         env: Environment.
#         config: Configuration dictionary.
#         num_eval_episodes: Number of episodes to evaluate the agent.
#         num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
#         video_frame_skip: Number of frames to skip between renders.
#         eval_temperature: Action sampling temperature.

#     Returns:
#         A tuple containing the statistics, trajectories, and rendered videos.
#     """
#     actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
#     trajs = []
#     stats = defaultdict(list)

#     renders = []
#     for i in trange(num_eval_episodes + num_video_episodes):
#         traj = defaultdict(list)
#         should_render = i >= num_eval_episodes

#         observation, info = env.reset()
#         done = False
#         step = 0
#         render = []
#         while not done:
#             action = actor_fn(observations=observation, temperature=eval_temperature)
#             action = np.array(action)
#             action = np.clip(action, -1, 1)

#             try:
#                 next_observation, reward, terminated, truncated, info = env.step(action)
#             except Exception as e:
#                 print("[EVAL] physics error -> early reset:", repr(e))
#                 try:
#                     next_observation, info = env.reset()
#                 except Exception as e2:
#                     print("[EVAL] reset failed:", repr(e2))
#                     break #end current traj.
#                 terminated, truncated = True, True

#             done = terminated or truncated
#             step += 1

#             if should_render and (step % video_frame_skip == 0 or done):
#                 frame = env.render().copy()
#                 render.append(frame)

#             transition = dict(
#                 observation=observation,
#                 next_observation=next_observation,
#                 action=action,
#                 reward=reward,
#                 done=done,
#                 info=info,
#             )
#             add_to(traj, transition)
#             observation = next_observation
#         if i < num_eval_episodes:
#             add_to(stats, flatten(info))
#             trajs.append(traj)
#         else:
#             renders.append(np.array(render))

#     for k, v in stats.items():
#         stats[k] = np.mean(v)

#     return stats, trajs, renders

def evaluate(
    agent, env, config=None,
    num_eval_episodes=50, num_video_episodes=0,
    video_frame_skip=3, eval_temperature=0,
):
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs, renders = [], []
    stats = defaultdict(list)

    img_hw = tuple((config or {}).get('img_hw', (240, 320)))

    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        observation, info = env.reset()
        observation = _standardize_observation(observation, img_hw)
        done = False
        step, render = 0, []

        while not done:
            action = actor_fn(observations=observation, temperature=eval_temperature)
            action = np.asarray(action, dtype=np.float32)
            if action.ndim > 1:
                action = action[0]          # (1, A) -> (A,)
            action = np.clip(action, -1.0, 1.0)

            reward = 0.0
            next_observation = observation
            terminated = truncated = False
            step_info = {}

            try:
                next_observation, reward, terminated, truncated, step_info = env.step(action)
            except Exception as e:
                print("[EVAL] physics error -> early reset:", repr(e))
                try:
                    next_observation, step_info = env.reset()
                except Exception as e2:
                    print("[EVAL] reset failed:", repr(e2))
                    terminated, truncated = True, True

            info = step_info
            next_observation = _standardize_observation(next_observation, img_hw)
            done = terminated or truncated
            step += 1

            if should_render and (step % video_frame_skip == 0 or done):
                try:
                    frame = env.render().copy()
                    render.append(frame)
                except Exception:
                    pass 

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=float(reward),
                done=bool(done),
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.asarray(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
