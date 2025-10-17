

# load_parquet.py
import io
import os
import glob
import numpy as np

# parquet 읽기용 (optional)
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


# 꼭 추가
import io, base64, re
import numpy as np

def _decode_png_bytes_array(pyarrow_binary_array, target_hw=None):
    """
    PyArrow 컬럼(원소가 dict/list/bytes/str 등 뒤죽박죽일 수 있음) -> (N,H,W,C) uint8
    target_hw: (H, W) 지정 시 리사이즈( Pillow 사용 시 )
    """
    try:
        from PIL import Image
        use_pil = True
    except Exception:
        use_pil = False
        import imageio.v2 as iio  # type: ignore

    def _extract_bytes(x):
        """원소 x에서 PNG bytes를 최대한 뽑아내기."""
        if x is None:
            return None
        # 직접 bytes 계열
        if isinstance(x, (bytes, bytearray, memoryview)):
            return bytes(x)
        # dict이면 값들 중에서 bytes 후보 탐색
        if isinstance(x, dict):
            # 키 이름 단서: 'bytes', 'data', 'png', '0', 0 등...
            # 일단 모든 value를 순회하며 첫 bytes-like 추출
            for v in x.values():
                b = _extract_bytes(v)
                if b is not None:
                    return b
            return None
        # 리스트/튜플이면 원소 순회
        if isinstance(x, (list, tuple)):
            for v in x:
                b = _extract_bytes(v)
                if b is not None:
                    return b
            return None
        # 문자열이면 base64 data URL 또는 hex일 가능성 처리
        if isinstance(x, str):
            s = x.strip()
            # data URL (base64)
            if s.startswith("data:image"):
                try:
                    b64 = s.split(",", 1)[1]
                    return base64.b64decode(b64)
                except Exception:
                    pass
            # HEX(예: "89504E47..." = PNG Signature)
            if len(s) % 2 == 0 and re.fullmatch(r"[0-9A-Fa-f]+", s or ""):
                try:
                    return bytes.fromhex(s)
                except Exception:
                    pass
            # 그 외는 패스
            return None
        # 모르는 타입은 패스
        return None

    data = pyarrow_binary_array.to_pylist()
    imgs = []
    for item in data:
        b = _extract_bytes(item)
        if b is None:
            imgs.append(None)
            continue
        buf = io.BytesIO(b)
        if use_pil:
            from PIL import Image  # type: ignore
            im = Image.open(buf).convert("RGB")
            if target_hw is not None:
                im = im.resize((target_hw[1], target_hw[0]))  # (W,H)
            arr = np.asarray(im, dtype=np.uint8)
        else:
            import imageio.v2 as iio  # type: ignore
            arr = iio.imread(buf)
            if arr.ndim == 2:  # gray → RGB
                arr = np.repeat(arr[..., None], 3, axis=-1)
            arr = arr.astype(np.uint8)
        imgs.append(arr)

    first = next((x for x in imgs if x is not None), None)
    if first is None:
        raise ValueError("No valid image could be decoded from the column.")
    H, W, C = first.shape
    out = np.zeros((len(imgs), H, W, C), dtype=np.uint8)
    for i, x in enumerate(imgs):
        out[i] = first if x is None else x
    return out



def _as_numpy_array_col(series_or_chunked):
    """
    PyArrow(또는 pandas) 컬럼에서 row마다 list/ndarray가 들어있는 경우를 (N, ...) numpy로 변환.
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


def _load_aloha_parquet_dataset(root_dir: str, val_ratio: float = 0.1):
    """
    ALOHA parquet 로더
    - PNG bytes 컬럼을 NHWC(uint8) 이미지로 디코딩
    - state(벡터)와 image를 모두 포함한 Dict 관측을 구성: {"image": NHWC, "state": D}
    - 에피소드 단위로 next_observations 생성
    - train/val split 반환
    """
    if not _HAVE_ARROW:
        raise ImportError("pyarrow is required to load parquet datasets.")

    data_dir = os.path.join(root_dir, "data")
    files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {data_dir}")

    ds = pads.dataset(files, format="parquet")
    table = ds.to_table()
    print(table)

    # 컬럼명 매핑(소문자 → 원래명)
    cols = {name.lower(): name for name in table.column_names}
    def has(*cands):
        return next((cols[c] for c in cands if c in cols), None)

    # 스키마에 맞게 컬럼 탐색
    obs_state_col = has("observation.state", "observations", "obs", "state", "states")
    act_col       = has("action", "actions")
    done_col      = has("next.done", "dones", "done", "terminals", "terminal")
    rew_col       = has("rewards", "reward")
    ep_col        = has("episode_index", "episode", "episode_id", "traj_id")
    step_col      = has("frame_index", "t", "step", "time", "timestamp")

    # 이미지 컬럼 후보
    img_top_col   = has("observation.images.top", "images.top", "top_img", "rgb_top")
    img_wrist_col = has("observation.images.wrist", "images.wrist", "wrist_img", "rgb_wrist")

    if act_col is None:
        raise KeyError(f"'actions' column not found. Available: {list(cols.keys())}")
    if (obs_state_col is None) and (img_top_col is None) and (img_wrist_col is None):
        raise KeyError("No usable observation column found (neither state nor image columns).")

    # 간단 array 변환기 (list-of-array 형태 컬럼용)
    def _as_np_from_table(colname):
        data = table[colname].to_pylist()
        arr0 = np.asarray(data[0])
        out = np.empty((len(data),) + arr0.shape, dtype=arr0.dtype)
        for i, v in enumerate(data):
            out[i] = np.asarray(v)
        return out

    # --- 관측 구성: state, image (둘 중 일부만 있을 수도 있음)
    state_arr = _as_np_from_table(obs_state_col) if obs_state_col else None  # (N, D)
    # PNG bytes → NHWC(uint8)
    img_top   = _decode_png_bytes_array(table[img_top_col])   if img_top_col   else None
    img_wrist = _decode_png_bytes_array(table[img_wrist_col]) if img_wrist_col else None
    image_arr = img_top if img_top is not None else img_wrist  # (N, H, W, C) or None

    # actions / rewards / done / episode / step
    actions = _as_np_from_table(act_col)  # (N, A)
    if rew_col:
        rewards = _as_np_from_table(rew_col).squeeze(-1)
    else:
        rewards = np.zeros((len(actions),), dtype=np.float32)

    if done_col:
        try:
            terminals = np.asarray(table[done_col].to_numpy(zero_copy_only=False)).astype(bool)
        except Exception:
            terminals = np.asarray(table[done_col].to_pylist()).astype(bool)
    else:
        terminals = np.zeros((len(actions),), dtype=bool)

    episodes = np.asarray(table[ep_col].to_numpy(zero_copy_only=False)) if ep_col else np.zeros((len(actions),), dtype=np.int64)
    steps    = np.asarray(table[step_col].to_numpy(zero_copy_only=False)) if step_col else np.arange(len(actions))

    # ---- 정렬 (episode, step 기준)
    order = np.lexsort((steps, episodes))
    actions  = actions[order]
    rewards  = rewards[order]
    terminals = terminals[order]
    episodes = episodes[order]
    steps    = steps[order]
    if state_arr is not None:
        state_arr = state_arr[order]
    if image_arr is not None:
        image_arr = image_arr[order]

    # ---- 에피소드별로 next_obs 생성
    obs_state_list, next_state_list = [], []
    obs_image_list, next_image_list = [], []
    act_list, rew_list, term_list   = [], [], []

    uniq, starts = np.unique(episodes, return_index=True)
    ends = np.r_[starts[1:], len(episodes)]

    for s, e in zip(starts, ends):
        a  = actions[s:e]
        r  = rewards[s:e]
        d  = terminals[s:e].copy()

        if not d.any():
            d[-1] = True

        act_list.append(a)
        rew_list.append(r)
        term_list.append(d)

        # state
        if state_arr is not None:
            st = state_arr[s:e]                           # (T, D)
            nst = np.concatenate([st[1:], st[-1:]], 0)    # (T, D)
            obs_state_list.append(st)
            next_state_list.append(nst)

        # image
        if image_arr is not None:
            im = image_arr[s:e]                           # (T, H, W, C)
            nim = np.concatenate([im[1:], im[-1:]], 0)    # (T, H, W, C)
            obs_image_list.append(im)
            next_image_list.append(nim)

    actions           = np.concatenate(act_list, axis=0)          # (N, A)
    rewards           = np.concatenate(rew_list, axis=0)          # (N,)
    terminals         = np.concatenate(term_list, axis=0).astype(np.uint8)  # (N,)

    # 관측/넥스트 관측 Dict로 합치기
    observations_dict = {}
    next_observations_dict = {}
    if state_arr is not None:
        observations_dict["state"]      = np.concatenate(obs_state_list, axis=0).astype(np.float32)  # (N, D)
        next_observations_dict["state"] = np.concatenate(next_state_list, axis=0).astype(np.float32) # (N, D)
    if image_arr is not None:
        observations_dict["image"]      = np.concatenate(obs_image_list, axis=0).astype(np.uint8)    # (N, H, W, C)
        next_observations_dict["image"] = np.concatenate(next_image_list, axis=0).astype(np.uint8)   # (N, H, W, C)

    # ---- split
    N = len(actions)
    n_val = int(N * float(val_ratio))
    rng = np.random.RandomState(42)
    perm = rng.permutation(N)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def _slice(idx_):
        terms = terminals[idx_].astype(np.uint8)
        timeo = np.zeros_like(terms, dtype=np.uint8)
        masks = (1 - np.clip(terms | timeo, 0, 1)).astype(np.float32)

        obs_slice = {k: v[idx_] for k, v in observations_dict.items()}
        next_obs_slice = {k: v[idx_] for k, v in next_observations_dict.items()}

        return dict(
            observations      = obs_slice,
            actions           = actions[idx_],
            rewards           = rewards[idx_],
            terminals         = terminals[idx_],
            next_observations = next_obs_slice,
            masks             = masks,
        )

    train_dataset = _slice(train_idx)
    val_dataset   = _slice(val_idx) if n_val > 0 else None

    n_train = len(train_dataset["actions"])
    n_val   = 0 if val_dataset is None else len(val_dataset["actions"])
    obs_keys = list(train_dataset["observations"].keys())
    shapes = {k: train_dataset["observations"][k].shape for k in obs_keys}
    print(f"[ALOHA parquet] train={n_train}, val={n_val}")
    print(f"[ALOHA parquet] obs keys={obs_keys}, shapes={shapes}")
    print(f"[ALOHA parquet] actions={train_dataset['actions'].shape}")

    print("\n[DEBUG] ===== Dataset Type & Structure =====")
    print(f"train_dataset type: {type(train_dataset)}")
    for k, v in train_dataset.items():
        print(f"  {k:20s} : {type(v)}")
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, np.ndarray):
                    print(f"      {kk:16s} -> np.ndarray {vv.shape} {vv.dtype}")
                else:
                    print(f"      {kk:16s} -> {type(vv)}")
        elif isinstance(v, np.ndarray):
            print(f"      shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"      value={v}")

    print("[DEBUG] =====================================\n")

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
    # 재현성 위해 고정 시드 섞기
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