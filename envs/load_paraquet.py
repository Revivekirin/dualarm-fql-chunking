import os, glob
import numpy as np
from io import BytesIO
from PIL import Image
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.compute as pc
from collections import deque
import random
from typing import Dict, Iterator, Optional, Tuple, List

import cv2
from concurrent.futures import ThreadPoolExecutor


# ---------- PNG decode ----------
def _decode_png_to_array(byte_arr: bytes) -> np.ndarray:
    with Image.open(BytesIO(byte_arr)) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


# ---------- batched Arrow helpers ----------
def _iter_batches(ds: pads.Dataset, columns: List[str], batch_size: int = 65536):
    scanner = ds.scanner(columns=columns, batch_size=batch_size, use_threads=True)
    for rb in scanner.to_batches():
        yield rb


def _column_as_numpy(rb: pa.RecordBatch, name: str) -> np.ndarray:
    col = rb.column(rb.schema.get_field_index(name))
    try:
        return col.to_numpy(zero_copy_only=False)
    except Exception:
        return np.asarray(col.to_pylist())


# ---------- (ep, step) 정렬 인덱스 ----------
def _sorted_index(ds: pads.Dataset, ep_col: str, step_col: str) -> Tuple[np.ndarray, np.ndarray]:
    eps, steps, ptrs = [], [], []
    total = 0
    for rb in _iter_batches(ds, [ep_col, step_col], batch_size=200_000):
        e = _column_as_numpy(rb, ep_col).astype(np.int64)
        s = _column_as_numpy(rb, step_col).astype(np.int64)
        n = len(e)
        eps.append(e)
        steps.append(s)
        ptrs.append(np.arange(total, total + n, dtype=np.int64))
        total += n
    eps = np.concatenate(eps, 0)
    steps = np.concatenate(steps, 0)
    ptrs = np.concatenate(ptrs, 0)
    order = np.lexsort((steps, eps))
    return order, ptrs


# ---------- 전이 스트리머 ----------
class AlohaTransitionStreamer:
    """
    Parquet을 에피소드-스텝 정렬 순서로 순회하며 (s, a, r, s', done, mask) 전이를 생성.
    픽셀은 PNG 바이트를 '필요한 순간'에만 디코드.
    완전 스트리밍: 전체 N·H·W·C 배열 사전할당/메모리맵 없음.
    """

    def __init__(
        self,
        root_dir: str,
        use_pixels: bool = True,
        shuffle_buffer: int = 2048,
        seed: int = 42,
        batch_size: int = 128,
        default_batch_size: int = 256,
    ):
        self.root_dir = root_dir
        self.use_pixels = use_pixels
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = int(batch_size)
        self.rng = random.Random(seed)
        self.default_batch_size = int(default_batch_size)

        data_dir = os.path.join(root_dir, "data")
        self.files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
        if not self.files:
            raise FileNotFoundError(f"No parquet found under {data_dir}")
        self.ds = pads.dataset(self.files, format="parquet")

        names = self.ds.schema.names
        cols = {n.lower(): n for n in names}

        def has(*cands): return next((cols[c] for c in cands if c in cols), None)

        self.img_col = has("observation.images.top")
        self.state_col = has("observation.state", "observations", "obs", "state", "states")
        self.act_col = has("action", "actions")
        self.done_col = has("next.done", "dones", "done", "terminals", "terminal")
        self.rew_col = has("rewards", "reward")
        self.ep_col = has("episode_index", "episode", "episode_id", "traj_id")
        self.step_col = has("frame_index", "t", "step", "time", "timestamp")
        if not (self.state_col and self.act_col and self.ep_col and self.step_col):
            raise KeyError(f"Missing required columns. Have: {list(cols.keys())}")

        # 정렬용 인덱스
        self.order, self.row_ptrs = _sorted_index(self.ds, self.ep_col, self.step_col)

        # 에피소드 인덱스 전체 확보
        eps = []
        for rb in _iter_batches(self.ds, [self.ep_col], batch_size=200_000):
            eps.append(_column_as_numpy(rb, self.ep_col).astype(np.int64))
        self.ep_all = np.concatenate(eps, 0)[self.order]

        # ⚡ list 기반 셔플 버퍼 (O(B) random pop)
        self._buffer: list = []
        self._episode_cache: Dict[int, Dict[str, list]] = {}

    def __iter__(self):
        yield from self.stream_batches(batch_size=self.default_batch_size)

    def _push_episode_item(self, ep: int, item: dict):
        cache = self._episode_cache.setdefault(ep, dict(
            state=[], action=[], reward=[], done=[], pixels_top=[]
        ))
        cache["state"].append(item["state"])
        cache["action"].append(item["action"])
        cache["reward"].append(item["reward"])
        cache["done"].append(item["done"])
        if self.use_pixels:
            cache["pixels_top"].append(item["pixels_top"])


    def _flush_episode_to_transitions(self, ep: int):
        cache = self._episode_cache.pop(ep, None)
        if cache is None or len(cache["state"]) == 0:
            return

        def shift_next(arr_list):
            if len(arr_list) == 1:
                return [arr_list[0]]
            return arr_list[1:] + [arr_list[-1]]

        next_state = shift_next(cache["state"])
        if self.use_pixels:
            next_pixels = shift_next(cache["pixels_top"])

        for t in range(len(cache["state"])):
            obs = {"agent_pos": cache["state"][t]}
            nxt = {"agent_pos": next_state[t]}
            if self.use_pixels:
                obs.setdefault("pixels", {})["top"] = cache["pixels_top"][t]
                nxt.setdefault("pixels", {})["top"] = next_pixels[t]

            a = cache["action"][t]
            r = cache["reward"][t]
            d = bool(cache["done"][t])
            m = 0.0 if d else 1.0

            self._buffer.append((obs, a, r, np.uint8(d), nxt, np.float32(m)))

    def _random_pop_batch(self) -> Optional[dict]:
        if len(self._buffer) < self.batch_size:
            return None
        idxs = sorted(self.rng.sample(range(len(self._buffer)), self.batch_size), reverse=True)
        sel = []
        for i in idxs:
            sel.append(self._buffer.pop(i))

        def stack_agent(items):
            return np.stack(items, 0).astype(np.float32)

        obs_agent = stack_agent([b[0]["agent_pos"] for b in sel])
        next_agent = stack_agent([b[4]["agent_pos"] for b in sel])

        if self.use_pixels:
            obs_pixels = np.stack([b[0]["pixels"]["top"] for b in sel], 0).astype(np.uint8)
            next_pixels = np.stack([b[4]["pixels"]["top"] for b in sel], 0).astype(np.uint8)
            observations = {"pixels": {"top": obs_pixels}, "agent_pos": obs_agent}
            next_observations = {"pixels": {"top": next_pixels}, "agent_pos": next_agent}
        else:
            observations = obs_agent
            next_observations = next_agent

        actions = np.stack([b[1] for b in sel], 0).astype(np.float32)
        rewards = np.asarray([b[2] for b in sel], np.float32)
        terminals = np.asarray([b[3] for b in sel], np.uint8)
        masks = np.asarray([b[5] for b in sel], np.float32)

        return dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            next_observations=next_observations,
            masks=masks,
        )

    def _maybe_emit_batches(self) -> Iterator[dict]:
        while len(self._buffer) >= self.batch_size:
            batch = self._random_pop_batch()
            if batch is None:
                break
            yield batch

    def set_batch_size(self, bs: int):
        self.batch_size = int(bs)

    def stream_batches(self, batch_size: int | None = None) -> Iterator[dict]:
        if batch_size is not None:
            self.batch_size = int(batch_size)

        need_cols = [self.state_col, self.act_col, self.ep_col, self.step_col]
        if self.done_col:
            need_cols.append(self.done_col)
        if self.rew_col:
            need_cols.append(self.rew_col)
        if self.use_pixels:
            if self.img_col is None:
                raise KeyError("use_pixels=True 이지만 이미지 컬럼을 찾을 수 없습니다.")
            need_cols.append(self.img_col)

        N = len(self.order)
        inv = np.empty_like(self.order)
        inv[self.order] = np.arange(N, dtype=np.int64)
        offset = 0
        CHUNK = 50_000
        if self.use_pixels:
            CHUNK = 1024

        acc_sorted_pos: List[int] = []
        acc_items: List[Tuple[int, int, dict]] = []

        for rb in _iter_batches(self.ds, need_cols, batch_size=CHUNK):
            n = len(rb)
            glob_idx = np.arange(offset, offset + n, dtype=np.int64)
            where = inv[glob_idx]
            offset += n

            states = _column_as_numpy(rb, self.state_col)
            acts = _column_as_numpy(rb, self.act_col)
            eps = _column_as_numpy(rb, self.ep_col).astype(np.int64)
            steps = _column_as_numpy(rb, self.step_col).astype(np.int64)
            dones = _column_as_numpy(rb, self.done_col).astype(bool) if self.done_col else np.zeros((n,), bool)
            rews = _column_as_numpy(rb, self.rew_col).astype(np.float32) if self.rew_col else np.zeros((n,), np.float32)

            if self.use_pixels:
                bytes_ca = pc.struct_field(rb[self.img_col], "bytes")
                # bytes_ca = pc.struct_field(rb[self.img_col], "bytes")
                # bytelist = [bytes_ca[i].as_py() for i in range(n)]

                def _decode_resize_png_cv2(byte_arr: bytes, out_hw=(240, 320)) -> np.ndarray:
                    buf = np.frombuffer(byte_arr, dtype=np.uint8)
                    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if im is None:
                        raise ValueError("cv2.imdecode failed")
                    h, w = out_hw
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    return im

                # workers = max(4, (os.cpu_count() or 8))
                # with ThreadPoolExecutor(max_workers=workers) as ex:
                #     imgs = list(ex.map(_decode_resize_png_cv2, bytelist))
            else:
                bytes_ca=None
            
            img = None
            for i in range(n):
                payload = {
                    "state": states[i],
                    "action": acts[i],
                    "reward": rews[i],
                    "done": dones[i],
                }
                if self.use_pixels:
                    byte_arr = bytes_ca[i].as_py()
                    img = _decode_resize_png_cv2(byte_arr, out_hw=(240, 320))
                    payload["pixels_top"] = img

                acc_sorted_pos.append(int(where[i]))
                acc_items.append((int(eps[i]), int(steps[i]), payload))

            if len(acc_sorted_pos) >= CHUNK:
                yield from self._drain_acc(acc_sorted_pos, acc_items)
                acc_sorted_pos.clear()
                acc_items.clear()
                yield from self._maybe_emit_batches()

        if acc_sorted_pos:
            yield from self._drain_acc(acc_sorted_pos, acc_items)
            acc_sorted_pos.clear()
            acc_items.clear()

        for ep in list(self._episode_cache.keys()):
            self._flush_episode_to_transitions(ep)

        while True:
            batch = self._random_pop_batch()
            if batch is None:
                break
            yield batch

    def _drain_acc(self, acc_sorted_pos: List[int], acc_items: List[Tuple[int, int, dict]]) -> Iterator[dict]:
        if not acc_items:
            return
        order = np.argsort(np.asarray(acc_sorted_pos, dtype=np.int64))
        last_ep = None

        for j in order:
            ep, _step, payload = acc_items[j]
            last_ep = ep
            if not self.use_pixels and ("pixels_top" in payload):
                p2 = dict(payload)
                p2.pop("pixels_top", None)
                payload = p2

            if last_ep is not None:
                self._flush_episode_to_transitions(last_ep)
                yield from self._maybe_emit_batches()

            self._push_episode_item(ep, payload)
            if bool(payload.get("done", False)):
                self._flush_episode_to_transitions(ep)
                yield from self._maybe_emit_batches()
            last_ep = ep


def build_aloha_streamer(
    root_dir: str,
    batch_size: int = 128,
    shuffle_buffer: int = 2048,
    seed: int = 42,
    use_pixels: bool = True,
    default_batch_size: int = 256,
) -> AlohaTransitionStreamer:
    streamer = AlohaTransitionStreamer(
        root_dir=root_dir,
        use_pixels=use_pixels,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        batch_size=batch_size,
        default_batch_size=default_batch_size,
    )
    return streamer


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