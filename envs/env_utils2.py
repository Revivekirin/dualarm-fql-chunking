# envs/env_utils2.py
import collections
import re
import time
import os
import importlib
import glob
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace

import ogbench
from utils.datasets import Dataset

from envs.load_paraquet2 import _load_aloha_parquet_dataset, _load_aloha_scripted_dataset


class FlattenAlohaObs(gym.ObservationWrapper):
    """
    {"pixels": {"top": HWC uint8}, "agent_pos": D}  -->
    {"image": HWC uint8, "state": D float32}
    """
    def __init__(self, env, image_key=("pixels","top"), state_key="agent_pos"):
        super().__init__(env)
        self.image_key = image_key
        self.state_key = state_key

        obs_space = env.observation_space
        if not isinstance(obs_space, DictSpace):
            raise TypeError(f"Expect Dict obs, got {type(obs_space)}")

        # image subspace
        if image_key[0] not in obs_space.spaces:
            raise KeyError(f"Missing key '{image_key[0]}' in observation_space")
        if not isinstance(obs_space.spaces[image_key[0]], DictSpace):
            raise TypeError(f"'{image_key[0]}' must be a Dict subspace")
        img_dict_space = obs_space.spaces[image_key[0]]
        if image_key[1] not in img_dict_space.spaces:
            raise KeyError(f"Missing key '{image_key[0]}/{image_key[1]}' in observation_space")

        img_space = img_dict_space.spaces[image_key[1]]
        if not isinstance(img_space, Box):
            raise TypeError(f"Image subspace must be Box, got {type(img_space)}")

        if not (img_space.dtype == np.uint8 and len(img_space.shape) == 3):
            raise AssertionError(
                f"Image Box must be uint8 and HWC 3D. Got dtype={img_space.dtype}, shape={img_space.shape}"
            )

        if state_key not in obs_space.spaces:
            raise KeyError(f"Missing key '{state_key}' in observation_space")
        st_space = obs_space.spaces[state_key]
        if not isinstance(st_space, Box):
            raise TypeError(f"State subspace must be Box, got {type(st_space)}")

        self.observation_space = DictSpace({
            "image": img_space,
            "state": Box(
                low=st_space.low.astype(np.float32),
                high=st_space.high.astype(np.float32),
                shape=st_space.shape,
                dtype=np.float32
            ),
        })

    def observation(self, observation):
        img = observation[self.image_key[0]][self.image_key[1]]
        st  = observation[self.state_key].astype(np.float32)
        return {"image": img, "state": st}


class SafeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, max_norm_scale: float = 1.0):
        super().__init__(env)
        self.max_norm_scale = float(max_norm_scale)

    def action(self, action):
        a = np.asarray(action, dtype=np.float32)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(self.action_space, "low"):
            low, high = self.action_space.low, self.action_space.high
            a = np.clip(a, low, high)
        if self.max_norm_scale is not None:
            if hasattr(self.action_space, "high"):
                ref = np.mean(np.abs(self.action_space.high))
                ref = 1.0 if ref == 0 else ref
            else:
                ref = 1.0
            max_norm = self.max_norm_scale * ref * a.size**0.5
            nrm = np.linalg.norm(a)
            if nrm > max_norm and nrm > 0:
                a = a * (max_norm / nrm)
        return a


class EpisodeMonitor(gym.Wrapper):
    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += float(reward)
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        done = terminated or truncated
        if done:
            info['episode'] = {
                'final_reward': float(reward),
                'return': self.reward_sum,
                'length': self.episode_length,
                'duration': time.time() - self.start_time,
            }

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gym.Wrapper):
    """
    Dict/Box 모두 지원. 프레임스택은 필요한 경우에만 사용하세요.
    """
    def __init__(self, env, num_stack: int):
        super().__init__(env)
        assert num_stack >= 1
        self.num_stack = num_stack

        obs_space = self.observation_space
        self._is_dict = isinstance(obs_space, DictSpace)

        if not self._is_dict and not isinstance(obs_space, Box):
            raise TypeError(f"Unsupported observation space type: {type(obs_space)}")

        if self._is_dict:
            self.frames = {k: collections.deque(maxlen=num_stack) for k in obs_space.spaces.keys()}
            stacked_spaces = {}
            for k, sp in obs_space.spaces.items():
                if not isinstance(sp, Box):
                    raise TypeError(f"Dict subspace for key '{k}' must be Box, got {type(sp)}")
                low  = np.concatenate([sp.low]  * num_stack, axis=-1)
                high = np.concatenate([sp.high] * num_stack, axis=-1)
                stacked_spaces[k] = Box(low=low, high=high, dtype=sp.dtype)
            self.observation_space = DictSpace(stacked_spaces)
        else:
            self.frames = collections.deque(maxlen=num_stack)
            low  = np.concatenate([obs_space.low]  * num_stack, axis=-1)
            high = np.concatenate([obs_space.high] * num_stack, axis=-1)
            self.observation_space = Box(low=low, high=high, dtype=obs_space.dtype)

    def _stacked_obs(self):
        if self._is_dict:
            return {k: np.concatenate(list(self.frames[k]), axis=-1) for k in self.frames.keys()}
        else:
            return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        if self._is_dict:
            for k in self.frames.keys():
                self.frames[k].clear()
                for _ in range(self.num_stack):
                    self.frames[k].append(ob[k])
        else:
            self.frames.clear()
            for _ in range(self.num_stack):
                self.frames.append(ob)
        return self._stacked_obs(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        if self._is_dict:
            for k in self.frames.keys():
                self.frames[k].append(ob[k])
        else:
            self.frames.append(ob)
        return self._stacked_obs(), reward, terminated, truncated, info


def _clip_actions(dataset: dict, eps: float):
    clipped = dict(dataset)
    clipped["actions"] = np.clip(dataset["actions"], -1 + eps, 1 - eps)
    return clipped


def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5, aloha_task=None,
                          img_h=128, img_w=128):
    """
    변경점:
    - ALOHA parquet: 로더에서는 리사이즈하지 않음 (중복 방지)
    - 환경에서만 리사이즈(필요 시) 래핑
    - 오프라인/평가용 Dataset 객체만 반환
    """
    already_clipped = False

    if 'singletask' in env_name:
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)

    elif env_name == 'gym-aloha' or env_name.startswith('gym-aloha'):
        importlib.import_module("gym_aloha")  # ensure registration

        if aloha_task == "sim_insertion":
            env_id = os.environ.get("ALOHA_ENV_ID", "gym_aloha/AlohaInsertion-v0")
            dataset_root = os.environ.get("ALOHA_DATASET_DIR", "gym-aloha/aloha_sim_insertion_scripted_image")
        elif aloha_task == "sim_transfer":
            env_id = os.environ.get("ALOHA_ENV_ID", "gym_aloha/AlohaTransferCube-v0")
            dataset_root = os.environ.get("ALOHA_DATASET_DIR", "gym-aloha/aloha_sim_transfer_cube_scripted_image")
        else:
            raise ValueError("aloha_task must be one of {'sim_insertion','sim_transfer'}")

        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        import gymnasium
        env = gymnasium.make(env_id, render_mode="rgb_array")
        eval_env = gymnasium.make(env_id, render_mode="rgb_array")

        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])

        env = FlattenAlohaObs(env, image_key=("pixels","top"), state_key="agent_pos")
        eval_env = FlattenAlohaObs(eval_env, image_key=("pixels","top"), state_key="agent_pos")

        # 선택적 프레임스택 (메모리 고려: 기본 None 권장)
        if frame_stack is not None:
            env = FrameStackWrapper(env, frame_stack)
            eval_env = FrameStackWrapper(eval_env, frame_stack)

        env = SafeActionWrapper(env, max_norm_scale=1.0)
        eval_env = SafeActionWrapper(eval_env, max_norm_scale=1.0)

        # ---- load parquet (리사이즈 없음; 환경에서 맞추세요)
        train_dataset, val_dataset = _load_aloha_parquet_dataset(
            dataset_root, val_ratio=float(os.environ.get("ALOHA_VAL_RATIO", "0.1")), target_hw=None
        )

        # clip & masks 보장
        def _ensure_masks_any(ds):
            m = dict(ds)
            terms = m["terminals"].astype(np.uint8)
            m["masks"] = (1 - np.clip(terms, 0, 1)).astype(np.float32)
            return m

        if action_clip_eps is not None:
            train_dataset = _clip_actions(train_dataset, action_clip_eps)
            if val_dataset is not None:
                val_dataset = _clip_actions(val_dataset, action_clip_eps)
            already_clipped = True

        train_dataset = _ensure_masks_any(train_dataset)
        if val_dataset is not None:
            val_dataset = _ensure_masks_any(val_dataset)

        train_dataset = Dataset.create(**train_dataset)
        if val_dataset is not None:
            val_dataset = Dataset.create(**val_dataset)

    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    # 이 파일에서는 리사이즈 래퍼를 적용하지 않습니다.
    # (main2.py의 ResizeObsWrapper에서만 일관되게 적용)

    env.reset()
    eval_env.reset()

    # Dataset actions 추가 clip (이미 처리됐다면 생략)
    if action_clip_eps is not None and not already_clipped:
        if isinstance(train_dataset, dict):
            train_dataset = _clip_actions(train_dataset, action_clip_eps)
            if val_dataset is not None:
                val_dataset = _clip_actions(val_dataset, action_clip_eps)
        else:
            train_dataset = train_dataset.copy(
                add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )
            if val_dataset is not None:
                val_dataset = val_dataset.copy(
                    add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
                )
    return env, eval_env, train_dataset, val_dataset
