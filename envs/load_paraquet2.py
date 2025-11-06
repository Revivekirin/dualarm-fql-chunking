# envs/load_paraquet2.py
import os, io, glob, re, base64, itertools, inspect
import numpy as np

try:
    import pyarrow.dataset as pads
    import pyarrow as pa
    _HAVE_ARROW = True
except Exception:
    _HAVE_ARROW = False

try:
    import pandas as pd
    _HAVE_PANDAS = True
except Exception:
    _HAVE_PANDAS = False


def _safe_to_numpy(col):
    """Arrow Array/ChunkedArray -> numpy (zero_copy_only=False)"""
    try:
        return col.to_numpy(zero_copy_only=False)
    except Exception:
        return np.asarray(col.to_pylist())


def _extract_bytes_maybe(item):
    """여러 타입(dict/list/str/bytes)에서 PNG bytes를 최대한 꺼내기."""
    if item is None:
        return None
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    if isinstance(item, dict):
        for v in item.values():
            b = _extract_bytes_maybe(v)
            if b is not None:
                return b
        return None
    if isinstance(item, (list, tuple)):
        for v in item:
            b = _extract_bytes_maybe(v)
            if b is not None:
                return b
        return None
    if isinstance(item, str):
        s = item.strip()
        # data URL
        if s.startswith("data:image"):
            try:
                b64 = s.split(",", 1)[1]
                return base64.b64decode(b64)
            except Exception:
                pass
        # HEX
        if len(s) % 2 == 0 and re.fullmatch(r"[0-9A-Fa-f]+", s or ""):
            try:
                return bytes.fromhex(s)
            except Exception:
                pass
        return None
    return None


def _decode_png_bytes_batch(pylist_bytes):
    """
    pylist of {bytes/bytearray/memoryview/str/dict/...} -> list of NHWC uint8 arrays
    리사이즈는 여기서 하지 않습니다(중복 방지).
    """
    # PIL이 있으면 PIL, 아니면 imageio
    try:
        from PIL import Image
        use_pil = True
    except Exception:
        use_pil = False
        import imageio.v3 as iio  # type: ignore

    out = []
    for it in pylist_bytes:
        b = _extract_bytes_maybe(it)
        if b is None:
            out.append(None)
            continue
        bio = io.BytesIO(b)
        if use_pil:
            from PIL import Image  # type: ignore
            im = Image.open(bio).convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
        else:
            import imageio.v3 as iio  # type: ignore
            arr = iio.imread(bio)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=-1)
            arr = arr.astype(np.uint8)
        out.append(arr)
    return out


def _lower_name_map(names):
    return {n.lower(): n for n in names}


def _find_col(cols_map, *cands):
    return next((cols_map[c] for c in cands if c in cols_map), None)


def _as_np_from_arrow_col(col):
    """
    Arrow 컬럼이 list-of-array 같은 케이스일 때 pylist 경유 후 numpy로.
    """
    data = col.to_pylist()
    arr0 = np.asarray(data[0])
    out = np.empty((len(data),) + arr0.shape, dtype=arr0.dtype)
    for i, v in enumerate(data):
        out[i] = np.asarray(v)
    return out


def _finalize_next(obs_arr):
    """
    obs_arr(T, ...) -> next_arr(T, ...)  (마지막은 자기 자신 복사)
    메모리 복사를 최소화하는 방식으로 구현.
    """
    next_arr = np.empty_like(obs_arr)
    next_arr[:-1] = obs_arr[1:]
    next_arr[-1] = obs_arr[-1]
    return next_arr


def _slice_split(N, val_ratio, rng_seed=42):
    n_val = int(N * float(val_ratio))
    rng = np.random.RandomState(rng_seed)
    perm = rng.permutation(N)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return train_idx, val_idx


def _pack_train_val(observations, actions, rewards, terminals, next_observations, masks, idx):
    obs_slice = {k: v[idx] for k, v in observations.items()}
    next_obs_slice = {k: v[idx] for k, v in next_observations.items()}
    return dict(
        observations=obs_slice,
        actions=actions[idx],
        rewards=rewards[idx],
        terminals=terminals[idx],
        next_observations=next_obs_slice,
        masks=masks[idx],
    )


def _ensure_masks(terminals, timeouts=None):
    terms = terminals.astype(np.uint8)
    if timeouts is None:
        timeouts = np.zeros_like(terms, dtype=np.uint8)
    return (1 - np.clip(terms | timeouts, 0, 1)).astype(np.float32)


def _as_numpy_array_col(series_or_chunked):
    """
    (호환 목적) pandas Series/list-of-arrays -> (N, ...) numpy
    """
    if _HAVE_PANDAS and isinstance(series_or_chunked, pd.Series):
        data = series_or_chunked.to_numpy()
    else:
        data = series_or_chunked.to_pylist()
    arr0 = np.asarray(data[0])
    shape = (len(data),) + arr0.shape
    out = np.empty(shape, dtype=arr0.dtype)
    for i, v in enumerate(data):
        out[i] = np.asarray(v)
    return out


def _iter_batches(ds, batch_size=8192):
    """
    PyArrow 버전 호환을 위한 Scanner.to_batches 래퍼.
    - 일부 버전: to_batches(batch_size=...)
    - 일부 버전: to_batches() (인자 없음)
    """
    scanner = ds.scanner()
    sig = inspect.signature(scanner.to_batches)
    try:
        if 'batch_size' in sig.parameters:
            return scanner.to_batches(batch_size=batch_size)
        else:
            return scanner.to_batches()
    except TypeError:
        # 최후의 호환성 안전망
        try:
            return scanner.to_batches(batch_size=batch_size)
        except TypeError:
            return scanner.to_batches()


def _load_aloha_parquet_dataset(root_dir: str, val_ratio: float = 0.1, target_hw=None):
    """
    ALOHA parquet 로더 (개선 버전)
    - Arrow streaming(to_batches) + 사전할당
    - PNG -> NHWC(uint8) 디코딩 (리사이즈는 하지 않음; 환경 래퍼에서 통일)
    - next_observations는 시프트 기반으로 한 번만 생성
    """
    if not _HAVE_ARROW:
        raise ImportError("pyarrow is required to load parquet datasets.")

    data_dir = os.path.join(root_dir, "data")
    files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")

    ds = pads.dataset(files, format="parquet")
    N = ds.count_rows()
    if N == 0:
        raise ValueError("Empty parquet dataset.")

    # 첫 배치에서 스키마/열명 파악
    batches_iter = _iter_batches(ds, batch_size=8192)
    try:
        first_batch = next(batches_iter)
    except StopIteration:
        raise ValueError("Dataset yielded no batches.")

    cols_map = _lower_name_map(first_batch.schema.names)

    obs_state_col = _find_col(cols_map, "observation.state", "observations", "obs", "state", "states")
    act_col       = _find_col(cols_map, "action", "actions")
    done_col      = _find_col(cols_map, "next.done", "dones", "done", "terminals", "terminal")
    rew_col       = _find_col(cols_map, "rewards", "reward")
    ep_col        = _find_col(cols_map, "episode_index", "episode", "episode_id", "traj_id")
    step_col      = _find_col(cols_map, "frame_index", "t", "step", "time", "timestamp")

    img_top_col   = _find_col(cols_map, "observation.images.top", "images.top", "top_img", "rgb_top")
    img_wrist_col = _find_col(cols_map, "observation.images.wrist", "images.wrist", "wrist_img", "rgb_wrist")

    if act_col is None:
        raise KeyError(f"'actions' column not found. Available: {list(cols_map.values())}")
    if (obs_state_col is None) and (img_top_col is None) and (img_wrist_col is None):
        raise KeyError("No usable observation column found (neither state nor image columns).")

    # 1) 길이만큼 사전할당
    actions = None
    rewards = np.zeros((N,), dtype=np.float32) if rew_col else None
    terminals = np.zeros((N,), dtype=np.uint8)
    episodes = np.zeros((N,), dtype=np.int64)
    steps    = np.arange(N, dtype=np.int64)

    state_arr = None
    image_arr = None

    # 2) first_batch 포함, 전체 배치 순회하며 채우기
    filled = 0
    for batch in itertools.chain([first_batch], batches_iter):
        act_chunk = _as_np_from_arrow_col(batch[act_col])
        rows = act_chunk.shape[0]
        if rows == 0:
            continue

        # 남은 슬롯 계산
        remaining = N - filled
        if remaining <= 0:
            break

        # rows가 remaining을 초과하면 모든 컬럼을 remaining에 맞춰 잘라서 넣기
        if rows > remaining:
            act_chunk = act_chunk[:remaining]
            rows = remaining

        if actions is None:
            actions = np.empty((N,) + act_chunk.shape[1:], dtype=act_chunk.dtype)
        actions[filled:filled+rows] = act_chunk

        if rew_col:
            rew_chunk = _as_np_from_arrow_col(batch[rew_col]).squeeze(-1)[:rows]
            rewards[filled:filled+rows] = rew_chunk

        if done_col:
            try:
                terms_chunk = batch[done_col].to_numpy(zero_copy_only=False).astype(bool)
            except Exception:
                terms_chunk = np.asarray(batch[done_col].to_pylist()).astype(bool)
            terminals[filled:filled+rows] = terms_chunk[:rows].astype(np.uint8)

        if ep_col:
            episodes[filled:filled+rows] = _safe_to_numpy(batch[ep_col]).astype(np.int64)[:rows]
        if step_col:
            steps[filled:filled+rows] = _safe_to_numpy(batch[step_col]).astype(np.int64)[:rows]

        if obs_state_col:
            st_chunk = _as_np_from_arrow_col(batch[obs_state_col])[:rows]
            if state_arr is None:
                state_arr = np.empty((N,) + st_chunk.shape[1:], dtype=st_chunk.dtype)
            state_arr[filled:filled+rows] = st_chunk

        pick_col = img_top_col or img_wrist_col
        if pick_col:
            pylist = batch[pick_col].to_pylist()[:rows]
            dec = _decode_png_bytes_batch(pylist)
            if image_arr is None:
                first_valid = next((x for x in dec if x is not None), None)
                if first_valid is None:
                    raise ValueError("No valid image in the dataset.")
                H, W, C = first_valid.shape
                image_arr = np.zeros((N, H, W, C), dtype=np.uint8)
            for i, arr in enumerate(dec):
                image_arr[filled + i] = first_valid if arr is None else arr

        filled += rows

    # 3) 정렬 (episode, step)
    order = np.lexsort((steps, episodes))
    actions   = actions[order]
    terminals = terminals[order]
    episodes  = episodes[order]
    steps     = steps[order]
    if rew_col:
        rewards = rewards[order]
    else:
        rewards = np.zeros((len(actions),), dtype=np.float32)
    if state_arr is not None:
        state_arr = state_arr[order]
    if image_arr is not None:
        image_arr = image_arr[order]

    # 4) next 관측 생성 (한 번만, 시프트 기반)
    observations_dict = {}
    next_observations_dict = {}
    if state_arr is not None:
        observations_dict["state"]      = state_arr.astype(np.float32)
        next_observations_dict["state"] = _finalize_next(state_arr).astype(np.float32)
    if image_arr is not None:
        observations_dict["image"]      = image_arr.astype(np.uint8)
        next_observations_dict["image"] = _finalize_next(image_arr).astype(np.uint8)

    # 5) 마지막 전이는 항상 done=True 보장 (에피소드 경계)
    if not terminals.any():
        uniq, starts = np.unique(episodes, return_index=True)
        ends = np.r_[starts[1:], len(episodes)]
        for s, e in zip(starts, ends):
            terminals[e - 1] = 1

    masks = _ensure_masks(terminals, timeouts=None)

    # 6) train/val split
    N = len(actions)
    train_idx, val_idx = _slice_split(N, val_ratio)
    train_dataset = _pack_train_val(observations_dict, actions, rewards, terminals, next_observations_dict, masks, train_idx)
    val_dataset   = _pack_train_val(observations_dict, actions, rewards, terminals, next_observations_dict, masks, val_idx) if len(val_idx) > 0 else None

    print(f"[ALOHA parquet] rows={N}, obs keys={list(observations_dict.keys())}, "
          f"img={('image' in observations_dict)}, state={('state' in observations_dict)}")
    return train_dataset, val_dataset


def _load_aloha_scripted_dataset(root_dir: str, val_ratio: float = 0.1):
    """
    기존 npz/npy 로더 (큰 변경 없음) — next_obs는 로더에서 1회 생성.
    """
    assert os.path.isdir(root_dir), f"Dataset dir not found: {root_dir}"

    cand = sorted(
        glob.glob(os.path.join(root_dir, "*.npz"))
        + glob.glob(os.path.join(root_dir, "*.npy"))
        + glob.glob(os.path.join(root_dir, "episode_*.npz"))
    )
    if not cand:
        raise FileNotFoundError(
            f"No dataset files (*.npz/*.npy) found under {root_dir}"
        )

    episodes = []
    for fp in cand:
        data = np.load(fp, allow_pickle=True)
        if hasattr(data, "files"):
            keys = set(data.files)
            arr = {k: data[k] for k in data.files}
        else:
            arr = dict(data.item()) if isinstance(data.item(), dict) else {"array": data}
            keys = set(arr.keys())

        obs_key = "observations" if "observations" in keys else ("states" if "states" in keys else None)
        act_key = "actions" if "actions" in keys else None
        rew_key = "rewards" if "rewards" in keys else ("reward" if "reward" in keys else None)
        done_key = "dones" if "dones" in keys else ("terminals" if "terminals" in keys else ("done" if "done" in keys else None))

        if obs_key is None or act_key is None:
            raise KeyError(f"{fp}: cannot find required keys (observations/states, actions). Found: {sorted(keys)}")

        obs = np.asarray(arr[obs_key])
        act = np.asarray(arr[act_key])
        rew = np.asarray(arr[rew_key]) if rew_key else np.zeros((len(act),), dtype=np.float32)
        done = np.asarray(arr[done_key]) if done_key else np.zeros((len(act),), dtype=bool)

        T = min(len(obs), len(act), len(rew), len(done))
        obs, act, rew, done = obs[:T], act[:T], rew[:T], done[:T]
        episodes.append({"observations": obs, "actions": act, "rewards": rew, "dones": done})

    obs_list, act_list, rew_list, done_list, next_obs_list = [], [], [], [], []
    for ep in episodes:
        o, a, r, d = ep["observations"], ep["actions"], ep["rewards"], ep["dones"]
        no = np.empty_like(o)
        no[:-1] = o[1:]
        no[-1] = o[-1]
        d = d.copy()
        d[-1] = True
        obs_list.append(o); act_list.append(a); rew_list.append(r); done_list.append(d); next_obs_list.append(no)

    observations = np.concatenate(obs_list, axis=0)
    actions = np.concatenate(act_list, axis=0)
    rewards = np.concatenate(rew_list, axis=0)
    terminals = np.concatenate(done_list, axis=0).astype(np.uint8)
    next_observations = np.concatenate(next_obs_list, axis=0)
    masks = _ensure_masks(terminals)

    N = len(actions)
    train_idx, val_idx = _slice_split(N, val_ratio)
    def _slice(idx):
        return dict(
            observations=observations[idx],
            actions=actions[idx],
            rewards=rewards[idx],
            terminals=terminals[idx],
            next_observations=next_observations[idx],
            masks=masks[idx],
        )
    train_dataset = _slice(train_idx)
    val_dataset   = _slice(val_idx) if len(val_idx) > 0 else None
    return train_dataset, val_dataset

