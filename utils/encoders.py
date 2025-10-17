import functools
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP
from flax.core import FrozenDict 

class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
        )(x)

        if self.max_pooling:
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
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out



class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""
    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = False
    # (선택) 관측 키를 명시하고 싶으면 config에서 넘겨도 됨
    image_keys: tuple[str, ...] = ("image", "pixels", "rgb", "images.top")
    state_keys: tuple[str, ...] = ("state", "proprio", "agent_pos")

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    def _extract_from_obs(self, obs):
        """dict/FrozenDict 관측에서 (image, state) 추출."""
        img = None
        st  = None
        for k in self.image_keys:
            if k in obs:
                img = obs[k]
                break
        if img is None:
            for v in (obs.values() if isinstance(obs, dict) else obs.keys()):
                try:
                    vv = obs[v] if not isinstance(obs, dict) else v
                except Exception:
                    vv = v
                arr = vv
                if hasattr(arr, "ndim") and arr.ndim >= 3 and arr.shape[-1] in (1,3,4):
                    img = arr
                    break
        for k in self.state_keys:
            if isinstance(obs, (dict, FrozenDict)) and k in obs:
                st = obs[k]
                break
        return img, st

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
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

        x = x.astype(jnp.float32) / 255.0  

        conv_out = x
        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        # (..., H, W, C) -> (..., F)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)
        return out



encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
}
