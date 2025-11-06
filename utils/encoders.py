import functools
from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from utils.networks import MLP
from flax.core import FrozenDict

Precision = jax.lax.Precision


class ResnetStack(nn.Module):
    """ResNet stack module (memory-friendly)."""
    num_features: int
    num_blocks: int
    max_pooling: bool = True

    # 추가: 정밀도/자료형 제어
    param_dtype: jnp.dtype = jnp.float32
    compute_dtype: jnp.dtype = jnp.bfloat16
    conv_precision: Precision | None = Precision.HIGH  # 안정성↑

    @nn.compact
    def __call__(self, x, *, train: bool = True):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
            param_dtype=self.param_dtype,
            dtype=self.compute_dtype,
            precision=self.conv_precision,
        )(x)

        if self.max_pooling:
            # max_pool은 dtype 인자를 받지 않지만 입력 dtype을 그대로 따라감
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
                param_dtype=self.param_dtype,
                dtype=self.compute_dtype,
                precision=self.conv_precision,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
                param_dtype=self.param_dtype,
                dtype=self.compute_dtype,
                precision=self.conv_precision,
            )(conv_out)
            conv_out = conv_out + block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder (with GAP + mixed precision)."""
    width: int = 1
    stack_sizes: Tuple[int, ...] = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float | None = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False

    # 추가: 출력 축약 방식 및 정밀도 제어
    global_avg_pool: bool = True         # True면 GAP, False면 flatten
    param_dtype: jnp.dtype = jnp.float32
    compute_dtype: jnp.dtype = jnp.bfloat16   # conv/act에 사용
    mlp_dtype: jnp.dtype = jnp.float32        # MLP는 안정성 위해 fp32
    conv_precision: Precision | None = Precision.HIGH

    image_keys: Tuple[str, ...] = ("image", "pixels", "rgb", "images.top")
    state_keys: Tuple[str, ...] = ("state", "proprio", "agent_pos")

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
                param_dtype=self.param_dtype,
                compute_dtype=self.compute_dtype,
                conv_precision=self.conv_precision,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    def _extract_from_obs(self, obs):
        """extract (image, state) from dict/FrozenDict."""
        img = None
        st = None
        # 우선 image_keys
        if isinstance(obs, (dict, FrozenDict)):
            for k in self.image_keys:
                if k in obs:
                    img = obs[k]
                    break
            # 후순위: 자동 감지 (H,W,C 형태)
            if img is None:
                for k, v in (obs.items() if isinstance(obs, dict) else obs.items()):
                    arr = v
                    if hasattr(arr, "ndim") and arr.ndim >= 3 and arr.shape[-1] in (1, 3, 4):
                        img = arr
                        break
            for k in self.state_keys:
                if k in obs:
                    st = obs[k]
                    break
        else:
            # tensor만 온 경우
            img = obs
        return img, st

    @nn.compact
    def __call__(self, x, train: bool = True, cond_var=None):
        # 입력 파싱
        state_feat = None
        if isinstance(x, (dict, FrozenDict)):
            img, st = self._extract_from_obs(x)
            if img is None:
                raise KeyError(
                    f"ImpalaEncoder: image tensor not found in observation keys. "
                    f"Tried {self.image_keys}."
                )
            x = img
            if st is not None:
                if hasattr(st, "dtype") and st.dtype != jnp.float32:
                    st = st.astype(jnp.float32)
                state_feat = st

        # [0,255] → [0,1], conv용 compute_dtype로 캐스팅
        x = x.astype(self.compute_dtype) / jnp.array(255.0, dtype=self.compute_dtype)

        conv_out = x
        for block in self.stack_blocks:
            conv_out = block(conv_out, train=train)
            if self.dropout_rate is not None:
                conv_out = nn.Dropout(rate=self.dropout_rate)(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm(dtype=self.compute_dtype)(conv_out)

        # 출력 축약: GAP 권장(메모리/파라미터 절감)
        if self.global_avg_pool:
            # NHWC 기준: H,W 축 평균
            out = jnp.mean(conv_out, axis=(-3, -2))  # (..., C)
        else:
            out = conv_out.reshape((*x.shape[:-3], -1))

        # MLP 전에는 안정성 위해 fp32로
        # conv/GAP 출력은 bfloat16 → MLP 입력 전엔 fp32로
        out = out.astype(jnp.float32)
        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}
