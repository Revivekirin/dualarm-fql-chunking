# data/replay_buffer.py
import collections
from typing import Any, Optional, Iterable, Union

import gymnasium as gym
import jax
import numpy as np
from data.dataset import Dataset, DatasetDict


def _init_replay_dict(obs_space: gym.Space, capacity: int) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int):
        if isinstance(dataset_dict, np.ndarray):
            dataset_dict[insert_index] = data_dict
        elif isinstance(dataset_dict, dict):
            for k in dataset_dict.keys():
                _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
        else:
            raise TypeError()


def _insert_batch_recursively(dst: DatasetDict, src: DatasetDict, slc):
    if isinstance(dst, np.ndarray):
        dst[slc] = src
    elif isinstance(dst, dict):
        for k in dst.keys():
            _insert_batch_recursively(dst[k], src[k], slc)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        reward_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        include_next_actions: Optional[bool] = False,
        include_label: Optional[bool] = False,
        include_grasp_penalty: Optional[bool] = False,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity, *reward_space.shape), dtype=reward_space.dtype),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
        )

        if include_next_actions:
            dataset_dict['next_actions'] = np.empty((capacity, *action_space.shape), dtype=action_space.dtype)
            dataset_dict['next_intvn'] = np.empty((capacity,), dtype=bool)

        if include_label:
            dataset_dict['labels'] = np.empty((capacity,), dtype=int)

        if include_grasp_penalty:
            dataset_dict['grasp_penalty'] = np.empty((capacity,), dtype=np.float32)

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    # 새로 추가: 벡터화된 배치 삽입 (슬라이스로 한 번에)
    def insert_batch(self, data_batch: DatasetDict):
        # data_batch의 길이 추정
        if isinstance(data_batch['actions'], np.ndarray):
            B = data_batch['actions'].shape[0]
        else:
            raise ValueError("data_batch must be a batch of transitions with numpy arrays.")

        end = self._insert_index + B
        if end <= self._capacity:
            slc = slice(self._insert_index, end)
            _insert_batch_recursively(self.dataset_dict, data_batch, slc)
        else:
            # 래핑(끝+처음) 두 번에 나눠 삽입
            first = self._capacity - self._insert_index
            _insert_batch_recursively(self.dataset_dict, {k: (v[:first] if isinstance(v, np.ndarray) else
                                                             {kk: vv[:first] for kk, vv in v.items()})
                                                         for k, v in data_batch.items()},
                                      slice(self._insert_index, self._capacity))
            remain = B - first
            _insert_batch_recursively(self.dataset_dict, {k: (v[first:] if isinstance(v, np.ndarray) else
                                                             {kk: vv[first:] for kk, vv in v.items()})
                                                         for k, v in data_batch.items()},
                                      slice(0, remain))

        self._insert_index = (self._insert_index + B) % self._capacity
        self._size = min(self._size + B, self._capacity)

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}, device=None):
        queue = collections.deque()
        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data, device=device))
        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def download(self, from_idx: int, to_idx: int):
        indices = np.arange(from_idx, to_idx)
        data_dict = self.sample(batch_size=len(indices), indx=indices)
        return to_idx, data_dict

    def get_download_iterator(self):
        last_idx = 0
        while True:
            if last_idx >= self._size:
                raise RuntimeError(f"last_idx {last_idx} >= self._size {self._size}")
            last_idx, batch = self.download(last_idx, self._size)
            yield batch
