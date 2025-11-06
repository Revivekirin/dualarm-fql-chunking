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


def _make_param_labels(params):
    """Create a label pytree for optax.multi_transform.

    modules_* 키 기준으로 간단/안전 라벨링:
      - modules_critic → 'critic'
      - modules_target_critic → 'frozen'
      - modules_actor_onestep_flow → 'actor'
      - modules_actor_bc_flow → 'bc'
      - modules_actor_bc_flow_encoder → 'bc' (별도로 select 호출하므로 존재)
      - 나머지 → 'frozen'
    """
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


class FQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ----------------------------- Losses -----------------------------

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss (TD)."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def bc_flow_loss(self, batch, grad_params, rng):
        """Compute the BC flow-matching loss (for µθ only)."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        return bc_flow_loss, {'bc_flow_loss': bc_flow_loss}

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss (for µω only): -Q + α * Distill."""
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng = jax.random.split(rng)

        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        # distill target MUST be stop_grad 
        target_flow_actions = jax.lax.stop_gradient(
            self.compute_flow_actions(batch['observations'], noises=noises)
        )
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)

        # Distillation term (W2^2 upper bound in action space)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q term (maximize Q)
        actor_actions_clipped = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions_clipped)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-8))
            q_loss = lam * q_loss

        actor_loss = self.config['alpha'] * distill_loss + q_loss

        # Monitoring
        actions_eval = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions_eval - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }

    # ------------------------- Total loss & update -------------------------

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Sum of three losses; gradients are routed by multi_transform masks."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, bc_flow_rng, critic_rng = jax.random.split(rng, 4)

        critic_loss, critic_info   = self.critic_loss(batch, grad_params, critic_rng)
        bc_flow_loss, flow_info    = self.bc_flow_loss(batch, grad_params, bc_flow_rng)
        actor_loss, actor_info     = self.actor_loss(batch, grad_params, actor_rng)

        loss = critic_loss + actor_loss + bc_flow_loss

        info.update({f'critic/{k}': v for k, v in critic_info.items()})
        info.update({f'bc_flow/{k}': v for k, v in flow_info.items()})
        info.update({f'actor/{k}': v for k, v in actor_info.items()})
        return loss, info

    # def target_update(self, network, module_name):
    #     """Soft-update the target network (critic only)."""
    #     new_target_params = jax.tree_util.tree_map(
    #         lambda p, tp: p * self.config['tau'] + tp * (1.0 - self.config['tau']),
    #         network.params[f'modules_{module_name}'],
    #         network.params[f'modules_target_{module_name}'],
    #     )
    #     network.params[f'modules_target_{module_name}'] = new_target_params

    def target_update(self, network, module_name):
        """Soft-update the target network (critic only)."""
        params = flax.core.unfreeze(network.params)
        src = params[f'modules_{module_name}']
        tgt = params[f'modules_target_{module_name}']

        new_tgt = jax.tree.map(
            lambda p, tp: p * self.config['tau'] + tp * (1.0 - self.config['tau']),
            src, tgt
        )
        params[f'modules_target_{module_name}'] = new_tgt
        return network.replace(params=flax.core.freeze(params))


    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    # ---------------------------- Action apis ----------------------------

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        """Sample actions from the one-step policy."""
        action_seed, _ = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        """Compute actions from the BC flow model using the Euler method (teacher)."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_bc_flow_encoder')(observations)
        actions = noises
        # Euler steps
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
        return actions

    # ---------------------------- Factory ----------------------------

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new FQL agent with multi-transform optimizer."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # Encoders
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # Networks
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
        )
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']

        # target critic init (= critic)
        network_params = flax.core.unfreeze(network_params)
        network_params['modules_target_critic'] = network_params['modules_critic']
        network_params = flax.core.freeze(network_params)

        # ---- multi-transform optimizer ----
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

        # config finalize
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fql',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            critic_lr=3e-4,   
            actor_lr=3e-4,    
            bc_lr=3e-4,       
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,
            q_agg='mean',
            alpha=10.0,
            flow_steps=10,
            normalize_q_loss=False,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config