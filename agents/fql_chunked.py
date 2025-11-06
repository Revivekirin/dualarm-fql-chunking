# agents/fql.py  — QC-FQL (chunked) version
import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


# ------------------------- helpers -------------------------

def _make_param_labels(params):
    """Labels for optax.multi_transform based on modules_* keys."""
    labels = {}

    def as_label(tree, tag):
        return jax.tree_util.tree_map(lambda _: tag, tree)

    for k, v in params.items():
        if k == 'modules_critic':
            labels[k] = as_label(v, 'critic')
        elif k == 'modules_target_critic':
            labels[k] = as_label(v, 'frozen')
        elif k == 'modules_actor_onestep_flow':
            labels[k] = as_label(v, 'actor')
        elif k == 'modules_actor_bc_flow':
            labels[k] = as_label(v, 'bc')
        elif k == 'modules_actor_bc_flow_encoder':
            labels[k] = as_label(v, 'bc')
        else:
            labels[k] = as_label(v, 'frozen')
    return flax.core.freeze(labels)


def _flatten_actions(a_chunk):
    """(B, H, Da) -> (B, AH)"""
    a_chunk = jnp.asarray(a_chunk)
    B, H, Da = a_chunk.shape
    return a_chunk.reshape(B, H * Da)


def _unflatten_actions(a_flat, H, Da):
    """(B, AH) -> (B, H, Da)"""
    a_flat = jnp.asarray(a_flat)
    B = a_flat.shape[0]
    return a_flat.reshape(B, H, Da)


def _batch_size_from_tree(tree):
    leaf = jax.tree_util.tree_leaves(tree)[0]
    return int(jnp.asarray(leaf).shape[0])


# ------------------------- agent -------------------------

class FQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent with action chunking."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()  # FrozenDict

    def _ensure_batched_obs(self, observations):
        """ob_dims(예시 관측의 랭크)를 기준으로 단일 샘플이면 B=1 축을 붙인다."""
        ob_dims = self.config.get('ob_dims', ())
        # ob_dims가 tuple(예: (14,))이면 그 길이가 '기본 랭크'
        base_rank = len(ob_dims) if isinstance(ob_dims, (tuple, list)) else None

        def fix(x):
            x = jnp.asarray(x)
            # base_rank가 있으면 그와 동일한 랭크면 배치축 추가
            if base_rank is not None and x.ndim == base_rank:
                return x[None, ...]
            # base_rank가 불명확하면, "배치 축이 없다"고 판단될 때 추가(힐리스틱)
            if base_rank is None and x.ndim >= 1 and x.shape[0] != 1:
                # 상태형(14,) 같이 보이면 배치축 추가
                if x.ndim == 1 or (x.ndim in (3,) and x.shape[0] not in (1,)):
                    return x[None, ...]
            return x

        return jax.tree_util.tree_map(fix, observations)

    # ----------------------------- Losses -----------------------------


    def critic_loss(self, batch, grad_params, rng):
        """
        TD on chunks:
          target = r_t^h + (γ^h) * mask_h * Q̄(s_{t+h}, a_{t+h: t+2h-1})
        batch keys:
          observations            : (B, ...)=s_t
          actions_chunk           : (B, H, Da)
          rewards_h               : (B,)
          masks_h                 : (B,)
          next_observations_h     : (B, ...)=s_{t+H}
        """
        H = int(self.config['H'])
        Da = int(self.config['action_dim_single'])
        AH = int(self.config['action_dim'])
        gamma_h = float(self.config['discount']) ** H

        # current chunk (flatten)
        a_flat = _flatten_actions(batch['actions_chunk'])  # (B, AH)

        # next chunk from s_{t+H}
        rng, sample_rng = jax.random.split(rng)
        next_actions_flat = self.sample_actions(batch['next_observations_h'], seed=sample_rng)  # (B, AH)
        next_actions_flat = jnp.clip(next_actions_flat, -1, 1)

        next_qs = self.network.select('target_critic')(
            batch['next_observations_h'], actions=next_actions_flat
        )
        next_q = next_qs.min(axis=0) if self.config['q_agg'] == 'min' else next_qs.mean(axis=0)

        target_q = batch['rewards_h'] + gamma_h * batch['masks_h'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=a_flat, params=grad_params)
        loss = jnp.square(q - target_q).mean()

        return loss, {
            'critic_loss': loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def bc_flow_loss(self, batch, grad_params, rng):
        """Flow-matching loss on flattened chunk space AH."""
        B = _batch_size_from_tree(batch['observations'])
        H = int(self.config['H'])
        Da = int(self.config['action_dim_single'])
        AH = H * Da

        a_flat = _flatten_actions(batch['actions_chunk'])  # (B, AH)

        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x0 = jax.random.normal(x_rng, (B, AH))
        t = jax.random.uniform(t_rng, (B, 1))
        xt = (1.0 - t) * x0 + t * a_flat
        vel = a_flat - x0

        pred = self.network.select('actor_bc_flow')(batch['observations'], xt, t, params=grad_params)
        loss = jnp.mean((pred - vel) ** 2)
        return loss, {'bc_flow_loss': loss}

    def actor_loss(self, batch, grad_params, rng):
        """
        Student one-step policy loss on AH:
          L = α * || μψ(s,z) - stopgrad(fξ(s, z)) ||^2  - E[Q(s, clip(μψ(s,z)))]
        """
        B = _batch_size_from_tree(batch['observations'])
        H = int(self.config['H'])
        Da = int(self.config['action_dim_single'])
        AH = H * Da

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (B, AH))

        # teacher (stopgrad)
        target_flat = jax.lax.stop_gradient(self.compute_flow_actions(batch['observations'], noises))
        # student
        actor_flat = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)

        distill = jnp.mean((actor_flat - target_flat) ** 2)

        actor_clip = jnp.clip(actor_flat, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_clip)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-8))
            q_loss = lam * q_loss

        loss = self.config['alpha'] * distill + q_loss

        # monitoring vs ground-truth chunk
        a_flat_gt = _flatten_actions(batch['actions_chunk'])
        mse = jnp.mean((actor_clip - a_flat_gt) ** 2)

        return loss, {
            'actor_loss': loss,
            'distill_loss': distill,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }

    # ------------------------- Total loss & update -------------------------

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, bc_flow_rng, critic_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        bc_loss, flow_info = self.bc_flow_loss(batch, grad_params, bc_flow_rng)
        act_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)

        loss = critic_loss + bc_loss + act_loss

        info.update({f'critic/{k}': v for k, v in critic_info.items()})
        info.update({f'bc_flow/{k}': v for k, v in flow_info.items()})
        info.update({f'actor/{k}': v for k, v in actor_info.items()})
        return loss, info

    def target_update(self, network, module_name):
        """Soft-update target critic."""
        params = flax.core.unfreeze(network.params)
        src = params[f'modules_{module_name}']
        tgt = params[f'modules_target_{module_name}']

        new_tgt = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1.0 - self.config['tau']),
            src, tgt
        )
        params[f'modules_target_{module_name}'] = new_tgt
        return network.replace(params=flax.core.freeze(params))

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    # ---------------------------- Action APIs ----------------------------

    # @jax.jit
    # def sample_actions(self, observations, seed=None, temperature=1.0):
    #     """
    #     Return flattened chunk actions: (B, AH).
    #     Caller may unflatten to (B,H,Da) if needed.
    #     """
    #     if seed is None:
    #         seed = self.rng
    #     action_seed, _ = jax.random.split(seed)
    #     B = _batch_size_from_tree(observations)
    #     AH = int(self.config['action_dim'])
    #     noises = jax.random.normal(action_seed, (B, AH))
    #     actions = self.network.select('actor_onestep_flow')(observations, noises)
    #     return jnp.clip(actions, -1, 1)

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        if seed is None:
            seed = self.rng
        action_seed, _ = jax.random.split(seed)

        observations = self._ensure_batched_obs(observations)
        B = jax.tree_util.tree_leaves(observations)[0].shape[0]

        AH = int(self.config['action_dim'])  # H*Da
        noises = jax.random.normal(action_seed, (B, AH))
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        return jnp.clip(actions, -1, 1)      # (B, AH)


    # @jax.jit
    # def compute_flow_actions(self, observations, noises):
    #     """Euler integration in flattened chunk space (teacher)."""
    #     if self.config.get('encoder', None) is not None:
    #         observations = self.network.select('actor_bc_flow_encoder')(observations)
    #     actions = noises
    #     for i in range(self.config['flow_steps']):
    #         t = jnp.full((actions.shape[0], 1), i / self.config['flow_steps'])
    #         vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
    #         actions = actions + vels / self.config['flow_steps']
    #     return jnp.clip(actions, -1, 1)
    @jax.jit
    def compute_flow_actions(self, observations, noises):
        observations = self._ensure_batched_obs(observations)
        if self.config.get('encoder', None) is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((actions.shape[0], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        return jnp.clip(actions, -1, 1)      # (B, AH)


    # ---------------------------- Factory ----------------------------

    @classmethod
    def create(cls, seed, ex_observations, ex_actions_chunk, config):
        """
        ex_actions_chunk: (1, H, Da)
        All networks operate in flattened action space AH = H*Da.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # infer shapes
        H = int(config['chunk_len'])
        Da = int(config['action_dim_single'])
        assert ex_actions_chunk.shape[-2:] == (H, Da), \
            f"ex_actions must be (B,{H},{Da}); got {ex_actions_chunk.shape}"
        ex_actions_flat = _flatten_actions(ex_actions_chunk)  # (1, AH)
        AH = H * Da

        ex_times = ex_actions_flat[..., :1]   # dummy time example
        # observation dims kept for compatibility; not strictly used here
        try:
            ob_dims = ex_observations.shape[1:]
        except Exception:
            ob_dims = ()

        # Encoders
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Networks (in AH)
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=AH,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=AH,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions_flat)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions_flat)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions_flat, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions_flat)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']

        # target critic = critic
        params = flax.core.unfreeze(network_params)
        params['modules_target_critic'] = params['modules_critic']
        network_params = flax.core.freeze(params)

        # multi-transform optimizer
        labels = _make_param_labels(network_params)
        tx = optax.multi_transform(
            {
                'critic': optax.adam(config.get('critic_lr', config['lr'])),
                'actor':  optax.adam(config.get('actor_lr',  config['lr'])),
                'bc':     optax.adam(config.get('bc_lr',     config['lr'])),
                'frozen': optax.set_to_zero(),
            },
            labels
        )
        network = TrainState.create(network_def, network_params, tx=tx)

        # finalize config
        config['ob_dims'] = ob_dims
        config['H'] = H
        config['action_dim_single'] = Da
        config['action_dim'] = AH  # flattened dim used everywhere
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Default config for QC-FQL (chunked)."""
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fql_chunked',
            # observation dims filled at runtime
            ob_dims=ml_collections.config_dict.placeholder(list),

            # chunking
            chunk_len=50,            # H
            action_dim_single=14,    # Da

            # lr
            lr=3e-4,
            critic_lr=3e-4,
            actor_lr=3e-4,
            bc_lr=3e-4,

            batch_size=256,

            # networks
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,

            # RL
            discount=0.99,
            tau=0.005,
            q_agg='mean',         # or 'min'
            alpha=10.0,
            flow_steps=10,
            normalize_q_loss=False,

            # encoder (set to None for state-only)
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
