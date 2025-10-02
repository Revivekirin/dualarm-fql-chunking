import collections
import re
import time
import os

import gymnasium
import numpy as np
import ogbench
from gymnasium.spaces import Box

from utils.datasets import Dataset
import glob
import numpy as np
import gymnasium as gym

from envs.load_paraquet import _load_aloha_parquet_dataset, _load_aloha_scripted_dataset

from gymnasium.spaces import Box, Dict as DictSpace


import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

import numpy as np
import gymnasium as gym

class SafeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, max_norm_scale: float = 1.0):
        super().__init__(env)
        self.max_norm_scale = float(max_norm_scale)

    def action(self, action):
        a = np.asarray(action, dtype=np.float32)
        # 1) NaN/Inf 방지
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) 박스 범위로 클립
        if hasattr(self.action_space, "low"):
            low, high = self.action_space.low, self.action_space.high
            a = np.clip(a, low, high)

        # 3) soft clip (과도한 노름 완화)
        if self.max_norm_scale is not None:
            # 액션 범위의 평균 크기 추정
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


class SelectObsKey(gym.ObservationWrapper):
    """
    Pick a (possibly auto-detected) key from Dict observations and optionally
    drop to the last frame to match offline obs dim.
    """
    def __init__(self, env, key: str | None = None, last_frame_only: bool = True, num_stack: int | None = None,
                 candidates: list[str] | None = None, target_feat_dim: int | None = None):
        super().__init__(env)
        self.num_stack = num_stack
        self.last_frame_only = last_frame_only

        obs_space = self.env.observation_space
        if not isinstance(obs_space, DictSpace):
            raise TypeError("SelectObsKey expects a Dict observation space")

        self._candidates = candidates or [
            "observation.state", "state", "proprio", "observation", "obs",
            "agent_state", "robot_state"
        ]
        keys = list(obs_space.spaces.keys())

        # 1) 명시 key가 있으면 그것부터
        if key and key in obs_space.spaces:
            picked = key
        else:
            # 2) 후보 키 순회
            picked = next((k for k in self._candidates if k in obs_space.spaces), None)
            # 3) 폴백: Box 서브스페이스 중에서 (a) target_feat_dim과 맞는 것, (b) 아니면 최소 차원
            if picked is None:
                box_items = [(k, sp) for k, sp in obs_space.spaces.items() if isinstance(sp, Box)]
                if not box_items:
                    raise TypeError(f"No Box subspace found in Dict obs. Keys: {keys}")
                if target_feat_dim is not None:
                    def eff_feat(sp):
                        # 프레임스택 고려: 마지막 축을 num_stack으로 나눠서 1프레임 feature 추정
                        feat = sp.shape[-1]
                        return feat // self.num_stack if (self.num_stack and feat % self.num_stack == 0) else feat
                    match = [k for k, sp in box_items if eff_feat(sp) == target_feat_dim]
                    picked = match[0] if match else None
                if picked is None:
                    # 가장 작은 feature 마지막축을 선택 (일반적으로 proprio가 가장 작음)
                    picked = sorted(box_items, key=lambda kv: kv[1].shape[-1])[0][0]

        if picked not in obs_space.spaces:
            raise TypeError(f"SelectObsKey: could not pick a key. Available: {keys}")

        self.key = picked
        sub = obs_space.spaces[self.key]
        if not isinstance(sub, Box):
            raise TypeError(f"SelectObsKey: subspace for '{self.key}' must be Box, got {type(sub)}")

        # 관측공간 정의 (마지막 프레임만 사용할지 여부 반영)
        if self.last_frame_only and self.num_stack and sub.shape[-1] % self.num_stack == 0:
            feat = sub.shape[-1] // self.num_stack
            low  = sub.low[..., -feat:]
            high = sub.high[..., -feat:]
            self.observation_space = Box(low=low, high=high, dtype=sub.dtype)
        else:
            self.observation_space = sub

        print(f"[SelectObsKey] picked key: '{self.key}' | space shape={sub.shape}")

    def observation(self, observation):
        x = observation[self.key]
        if self.last_frame_only and self.num_stack and x.shape[-1] % self.num_stack == 0:
            feat = x.shape[-1] // self.num_stack
            x = x[..., -feat:]
        return x


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

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

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)

class FrameStackWrapper(gym.Wrapper):
    """
    Stack last `num_stack` observations along the last axis.
    Works for Box observations and Dict observations (per-key stacking).
    """

    def __init__(self, env, num_stack: int):
        super().__init__(env)
        assert num_stack >= 1
        self.num_stack = num_stack

        obs_space = self.observation_space
        self._is_dict = isinstance(obs_space, DictSpace)

        if not self._is_dict and not isinstance(obs_space, Box):
            raise TypeError(f"Unsupported observation space type: {type(obs_space)}")

        # Prepare frame buffers and stacked observation_space
        if self._is_dict:
            # one deque per key
            self.frames = {k: collections.deque(maxlen=num_stack) for k in obs_space.spaces.keys()}
            # build stacked Dict space
            stacked_spaces = {}
            for k, sp in obs_space.spaces.items():
                if not isinstance(sp, Box):
                    raise TypeError(f"Dict subspace for key '{k}' must be Box, got {type(sp)}")
                low  = np.concatenate([sp.low]  * num_stack, axis=-1)
                high = np.concatenate([sp.high] * num_stack, axis=-1)
                stacked_spaces[k] = Box(low=low, high=high, dtype=sp.dtype)
            self.observation_space = DictSpace(stacked_spaces)
        else:
            # single deque for Box space
            self.frames = collections.deque(maxlen=num_stack)
            low  = np.concatenate([obs_space.low]  * num_stack, axis=-1)
            high = np.concatenate([obs_space.high] * num_stack, axis=-1)
            self.observation_space = Box(low=low, high=high, dtype=obs_space.dtype)

    def _stacked_obs(self):
        if self._is_dict:
            assert all(len(deq) == self.num_stack for deq in self.frames.values())
            return {k: np.concatenate(list(self.frames[k]), axis=-1) for k in self.frames.keys()}
        else:
            assert len(self.frames) == self.num_stack
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

        # goal 정보가 ndarray면 동일하게 스택(선택적)
        if 'goal' in info and isinstance(info['goal'], np.ndarray):
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)

        return self._stacked_obs(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        if self._is_dict:
            for k in self.frames.keys():
                self.frames[k].append(ob[k])
        else:
            self.frames.append(ob)
        return self._stacked_obs(), reward, terminated, truncated, info

def _clip_actions(dataset: dict, action_clip_eps: float):
    clipped = dataset.copy()
    clipped["actions"] = np.clip(
        dataset["actions"], -1 + action_clip_eps, 1 - action_clip_eps
    )
    return clipped

def make_env_and_datasets(env_name, frame_stack=None, action_clip_eps=1e-5, aloha_task=None):
    """Make offline RL environment and datasets.

    Args:
        env_name: Name of the environment or dataset.
        frame_stack: Number of frames to stack.
        action_clip_eps: Epsilon for action clipping.

    Returns:
        A tuple of the environment, evaluation environment, training dataset, and validation dataset.
    """

    if 'singletask' in env_name:
        # OGBench.
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        train_dataset = Dataset.create(**train_dataset)
        val_dataset = Dataset.create(**val_dataset)
    elif 'antmaze' in env_name and ('diverse' in env_name or 'play' in env_name or 'umaze' in env_name):
        # D4RL AntMaze.
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None
    elif 'pen' in env_name or 'hammer' in env_name or 'relocate' in env_name or 'door' in env_name:
        # D4RL Adroit.
        import d4rl.hand_manipulation_suite  
        from envs import d4rl_utils

        env = d4rl_utils.make_env(env_name)
        eval_env = d4rl_utils.make_env(env_name)
        dataset = d4rl_utils.get_dataset(env, env_name)
        train_dataset, val_dataset = dataset, None

    elif env_name == 'gym-aloha' or env_name.startswith('gym-aloha'):
        import importlib, os
        importlib.import_module("gym_aloha")  # ensure registration

        if aloha_task=="sim_insertion":
            env_id = os.environ.get("ALOHA_ENV_ID", "gym_aloha/AlohaInsertion-v0")
            dataset_root = os.environ.get("ALOHA_DATASET_DIR", "gym-aloha/aloha_sim_insertion_scripted_image")
        elif aloha_task=="sim_transfer":
            env_id = os.environ.get("ALOHA_ENV_ID", "gym_aloha/AlohaTransferCube-v0")
            dataset_root=os.environ.get("ALOHA_DATASET_DIR", "gym-aloha/aloha_sim_transfer_cube_scripted_image")
        val_ratio = float(os.environ.get("ALOHA_VAL_RATIO", "0.1"))

        # headless
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

        # ---- make envs
        env = gymnasium.make(env_id, render_mode="rgb_array")
        eval_env = gymnasium.make(env_id, render_mode="rgb_array")

        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])

        ob_key = "agent_pos" 
        env = SelectObsKey(env, key=ob_key, last_frame_only=True, num_stack=frame_stack, target_feat_dim=14)
        eval_env = SelectObsKey(eval_env, key=ob_key, last_frame_only=True, num_stack=frame_stack, target_feat_dim=14)
        env = SafeActionWrapper(env, max_norm_scale=1.0)
        eval_env = SafeActionWrapper(eval_env, max_norm_scale=1.0)

        # ---- load parquet (dict)
        train_dataset, val_dataset = _load_aloha_parquet_dataset(dataset_root, val_ratio=val_ratio)

        # ---- clip actions (dict)
        if action_clip_eps is not None:
            def _clip_actions(ds: dict, eps: float):
                out = ds.copy()
                out["actions"] = np.clip(out["actions"], -1 + eps, 1 - eps)
                return out
            train_dataset = _clip_actions(train_dataset, action_clip_eps)
            if val_dataset is not None:
                val_dataset = _clip_actions(val_dataset, action_clip_eps)

        # ----------------- dataset helper -----------------
        try:
            from flax.core.frozen_dict import FrozenDict
        except Exception:
            FrozenDict = ()

        def _to_mutable_mapping(ds):
            if isinstance(ds, dict):
                return dict(ds)
            if FrozenDict and isinstance(ds, FrozenDict):
                return dict(ds.unfreeze())
            try:
                return dict(ds)
            except Exception:
                raise TypeError(f"Unsupported dataset mapping type: {type(ds)}")

        def _clip_actions_any(ds, eps: float):
            m = _to_mutable_mapping(ds)
            m["actions"] = np.clip(m["actions"], -1 + eps, 1 - eps)
            return m

        def _ensure_masks_any(ds):
            m = _to_mutable_mapping(ds)
            terms = m["terminals"].astype(np.uint8)
            timeo = m.get("timeouts", np.zeros_like(terms, dtype=np.uint8)).astype(np.uint8)
            m["masks"] = (1 - np.clip(terms | timeo, 0, 1)).astype(np.float32)
            return m
        # ---------------------------------------------------

        # 액션 클리핑
        if action_clip_eps is not None:
            train_dataset = _clip_actions_any(train_dataset, action_clip_eps)
            if val_dataset is not None:
                val_dataset = _clip_actions_any(val_dataset, action_clip_eps)

        # 마스크 보장
        train_dataset = _ensure_masks_any(train_dataset)
        if val_dataset is not None:
            val_dataset = _ensure_masks_any(val_dataset)

        # dict/FrozenDict -> Dataset 객체
        from utils.datasets import Dataset
        train_dataset = Dataset.create(**train_dataset)
        if val_dataset is not None:
            val_dataset = Dataset.create(**val_dataset)

    else:
        raise ValueError(f'Unsupported environment: {env_name}')

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None:
        if isinstance(train_dataset, dict):
            train_dataset = _clip_actions(train_dataset, action_clip_eps)
            if val_dataset is not None:
                val_dataset = _clip_actions(val_dataset, action_clip_eps)
        else:
            # D4RL Dataset 객체 등은 기존 코드 유지
            train_dataset = train_dataset.copy(
                add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )
            if val_dataset is not None:
                val_dataset = val_dataset.copy(
                    add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
                )


    return env, eval_env, train_dataset, val_dataset



