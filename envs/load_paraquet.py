import os

import numpy as np
import ogbench

from utils.datasets import Dataset
import glob
import numpy as np

# parquet 읽기용
try:
    import pyarrow.dataset as pads
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAVE_ARROW = True
except Exception:
    _HAVE_ARROW = False

try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False


def _as_numpy_array_col(series_or_chunked):
    """
    PyArrow(또는 pandas) 컬럼에서 row마다 list/ndarray가 들어있는 경우를
    (N, ...) numpy 로 깔끔히 변환.
    """
    if _HAVE_PANDAS and isinstance(series_or_chunked, pd.Series):
        data = series_or_chunked.to_numpy()
    else:
        # pyarrow Array -> python list
        data = series_or_chunked.to_pylist()
    # 각 원소가 list/np.ndarray 라고 가정하고 stack
    arr0 = np.asarray(data[0])
    shape = (len(data),) + arr0.shape
    out = np.empty(shape, dtype=arr0.dtype)
    for i, v in enumerate(data):
        out[i] = np.asarray(v)
    return out

import os, glob, numpy as np
import pyarrow.dataset as pads
import pyarrow as pa

def _load_aloha_parquet_dataset(root_dir: str, val_ratio: float = 0.1):
    data_dir = os.path.join(root_dir, "data")
    files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")

    ds = pads.dataset(files, format="parquet")
    table = ds.to_table()

    cols = {name.lower(): name for name in table.column_names}
    def has(*cands): return next((cols[c] for c in cands if c in cols), None)

    obs_col  = has("observation.state", "observations", "obs", "state", "states")
    act_col  = has("action", "actions")
    done_col = has("next.done", "dones", "done", "terminals", "terminal")
    rew_col  = has("rewards", "reward")  
    ep_col   = has("episode_index", "episode", "episode_id", "traj_id")
    step_col = has("frame_index", "t", "step", "time", "timestamp")

    if not obs_col or not act_col:
        raise KeyError(f"Required columns not found. Found: {list(cols.keys())}")

    def _as_np(colname):
        data = table[colname].to_pylist()
        arr0 = np.asarray(data[0])
        out = np.empty((len(data),)+arr0.shape, dtype=arr0.dtype)
        for i,v in enumerate(data):
            out[i] = np.asarray(v)
        return out

    obs = _as_np(obs_col)
    act = _as_np(act_col)
    rew = _as_np(rew_col).squeeze(-1) if rew_col else np.zeros((len(act),), dtype=np.float32)

    if done_col:
        try:
            done = np.asarray(table[done_col].to_numpy(zero_copy_only=False)).astype(bool)
        except Exception:
            done = np.asarray(table[done_col].to_pylist()).astype(bool)
    else:
        done = np.zeros((len(act),), dtype=bool)

    ep   = np.asarray(table[ep_col].to_numpy(zero_copy_only=False)) if ep_col else np.zeros((len(act),), dtype=np.int64)
    step = np.asarray(table[step_col].to_numpy(zero_copy_only=False)) if step_col else np.arange(len(act))

    # 정렬
    order = np.lexsort((step, ep))
    obs, act, rew, done, ep = obs[order], act[order], rew[order], done[order], ep[order]

    # 에피소드별 next_obs, terminals
    obs_list, act_list, rew_list, term_list, next_obs_list = [], [], [], [], []
    uniq, starts = np.unique(ep, return_index=True)
    ends = np.r_[starts[1:], len(ep)]
    for s,e in zip(starts, ends):
        o, a, r, d = obs[s:e], act[s:e], rew[s:e], done[s:e].copy()
        if not d.any(): d[-1] = True           # 마지막 전이 강제 terminal
        no = np.concatenate([o[1:], o[-1:]], axis=0)
        obs_list.append(o); act_list.append(a); rew_list.append(r); term_list.append(d); next_obs_list.append(no)

    observations      = np.concatenate(obs_list, axis=0)
    actions           = np.concatenate(act_list, axis=0)
    rewards           = np.concatenate(rew_list, axis=0)
    terminals         = np.concatenate(term_list, axis=0).astype(np.uint8)
    next_observations = np.concatenate(next_obs_list, axis=0)


    # split
    N = len(actions)
    n_val = int(N * float(val_ratio))
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def _slice(idx_):
        terms = terminals[idx_].astype(np.uint8)
        timeo = np.zeros_like(terms, dtype=np.uint8)
        masks = (1 - np.clip(terms | timeo, 0, 1)).astype(np.float32)
        return dict(
            observations      = observations[idx_],
            actions           = actions[idx_],
            rewards           = rewards[idx_],
            terminals         = terminals[idx_],
            # timeouts          = np.zeros_like(terminals[idx_]),
            next_observations = next_observations[idx_],
            masks             = masks,       
        )

    train_dataset = _slice(train_idx)
    val_dataset   = _slice(val_idx) if n_val > 0 else None

    print(f"[ALOHA parquet] train={len(train_dataset['actions'])}, val={0 if val_dataset is None else len(val_dataset['actions'])}")
    print(f"[ALOHA parquet] obs={train_dataset['observations'].shape}, act={train_dataset['actions'].shape}")

    return train_dataset, val_dataset  



def _load_aloha_scripted_dataset(
    root_dir: str,
    val_ratio: float = 0.1,
):
    """
    ALOHA scripted 데모(npz/npy)들을 읽어 통합한 뒤
    {observations, actions, rewards, dones, next_observations} 딕셔너리로 반환.
    파일이 여러 개인 경우 에피소드 단위로 연결하고 next_obs를 경계에서 끊어줍니다.
    """
    assert os.path.isdir(root_dir), f"Dataset dir not found: {root_dir}"

    # 후보 파일들 수집
    cand = sorted(
        glob.glob(os.path.join(root_dir, "*.npz"))
        + glob.glob(os.path.join(root_dir, "*.npy"))
        + glob.glob(os.path.join(root_dir, "episode_*.npz"))
    )
    if not cand:
        raise FileNotFoundError(
            f"No dataset files (*.npz/*.npy) found under {root_dir} "
            "(git clone 후 경로를 확인하거나 ALOHA_DATASET_DIR 을 지정하세요)"
        )

    episodes = []
    for fp in cand:
        if fp.endswith(".npz") or fp.endswith(".npy"):
            data = np.load(fp, allow_pickle=True)
            # npz일 때: data.files, npy일 때: 객체일 수도 있음
            if hasattr(data, "files"):
                keys = set(data.files)
                arr = {k: data[k] for k in data.files}
            else:
                # npy에 dict 저장된 경우
                arr = dict(data.item()) if isinstance(data.item(), dict) else {"array": data}
                keys = set(arr.keys())

            # 키 후보 매핑
            # obs/state
            obs_key = "observations" if "observations" in keys else ("states" if "states" in keys else None)
            act_key = "actions" if "actions" in keys else None
            rew_key = "rewards" if "rewards" in keys else ("reward" if "reward" in keys else None)
            done_key = (
                "dones" if "dones" in keys else ("terminals" if "terminals" in keys else ("done" if "done" in keys else None))
            )

            if obs_key is None or act_key is None:
                # 시각/고차원 dict(obs) 구조일 수도 있으니 최소 폴백: 전부 np.array 로 변환 시도
                raise KeyError(
                    f"{fp}: cannot find required keys (observations/states, actions). Found: {sorted(keys)}"
                )

            obs = np.asarray(arr[obs_key])
            act = np.asarray(arr[act_key])
            # reward/done 없으면 0/False로 생성
            rew = np.asarray(arr[rew_key]) if rew_key else np.zeros((len(act),), dtype=np.float32)
            done = np.asarray(arr[done_key]) if done_key else np.zeros((len(act),), dtype=bool)

            # 길이 정합성(관례상 obs: T x Obs, act: T x A)
            T = min(len(obs), len(act), len(rew), len(done))
            obs, act, rew, done = obs[:T], act[:T], rew[:T], done[:T]

            episodes.append({"observations": obs, "actions": act, "rewards": rew, "dones": done})
        else:
            continue

    # 에피소드 연결 + next_obs 생성
    obs_list, act_list, rew_list, done_list, next_obs_list = [], [], [], [], []
    for ep in episodes:
        o, a, r, d = ep["observations"], ep["actions"], ep["rewards"], ep["dones"]
        # next_obs: 같은 에피소드에서만 유효(마지막은 자기 자신으로 복사)
        no = np.concatenate([o[1:], o[-1:]], axis=0)
        # 마지막 전이는 done=True 보장
        d = d.copy()
        d[-1] = True

        obs_list.append(o)
        act_list.append(a)
        rew_list.append(r)
        done_list.append(d)
        next_obs_list.append(no)

    observations = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    rewards = np.concatenate(rew_list, axis=0)
    dones = np.concatenate(done_list, axis=0)
    next_observations = np.concatenate(next_obs_list, axis=0)

    # train/val split
    N = len(actions)
    n_val = int(N * float(val_ratio))
    idx = np.arange(N)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    def _slice(idx_):
        return dict(
            observations=observations[idx_],
            actions=actions[idx_],
            rewards=rewards[idx_],
            dones=dones[idx_],
            next_observations=next_observations[idx_],
        )

    train_dataset = _slice(train_idx)
    val_dataset = _slice(val_idx) if n_val > 0 else None
    return train_dataset, val_dataset