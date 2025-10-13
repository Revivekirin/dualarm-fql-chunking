import functools
from typing import Sequence, Any, Dict

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP

class ProprioEncoder(nn.Module):
    hidden_dims: Sequence[int] = (256,)
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        return MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(x)

class AlohaMultiModalEncoder(nn.Module):
    """pixels.top + agent_pos 를 받아 concat 임베딩을 반환."""
    impala_num_blocks: int = 1
    impala_stack_sizes: tuple = (16, 128, 128)
    impala_width: int = 1
    impala_layer_norm: bool = False
    impala_mlp_hidden: Sequence[int] = (512,)

    # proprio 인코더 설정
    proprio_hidden: Sequence[int] = (256,)
    proprio_layer_norm: bool = False

    @nn.compact
    def __call__(self, obs: Dict[str, Any], train: bool = True):
        img = obs["pixels"]["top"]
        proprio = obs["agent_pos"]
        print("[DEBUG] img.shape :", img.shape)
        print("[DEBUG] proprio.shape :", proprio.shape)

        # IMPALA
        impala = ImpalaEncoder(
            width=self.impala_width,
            stack_sizes=self.impala_stack_sizes,
            num_blocks=self.impala_num_blocks,
            mlp_hidden_dims=self.impala_mlp_hidden,
            layer_norm=self.impala_layer_norm,
        )
        f_img = impala(img, train=train)   # (B, F_img)

        # proprio MLP
        f_prop = ProprioEncoder(
            hidden_dims=self.proprio_hidden,
            layer_norm=self.proprio_layer_norm,
        )(proprio)                         # (B, F_prop)

        return jnp.concatenate([f_img, f_prop], axis=-1)


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

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm()(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm)(out)

        return out


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'aloha_mm': AlohaMultiModalEncoder, 
}
