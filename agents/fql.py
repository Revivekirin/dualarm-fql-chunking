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


def ensure_batched_with_spec(x, spec_ndim):
    x = jnp.asarray(x)
    if x.ndim == spec_ndim:            # e.g., (H,W,C) or (D,)
        return x[None, ...]            # -> (1, ...)
    elif x.ndim == spec_ndim + 1:      # already (B, ...)
        return x
    elif x.ndim > spec_ndim + 1:       # e.g., (T,B, ...) or (N_envs,B, ...)
        new_batch = int(jnp.prod(jnp.array(x.shape[: x.ndim - spec_ndim - 1])))
        tail_shape = x.shape[-(spec_ndim+1):]
        return x.reshape((new_batch,) + tail_shape)
    else:
        return x[None, ...]
    
def _ensure_batched_tree(obs, base_ndims):
    # obs: nested dict/array (관측 트리)
    # base_ndims: nested dict/int (예시 배치에서 기록한 랭크 스펙)
    if isinstance(obs, dict):
        out = {}
        for k, v in obs.items():
            sub_spec = None
            if isinstance(base_ndims, dict) and (k in base_ndims):
                sub_spec = base_ndims[k]
            out[k] = _ensure_batched_tree(v, sub_spec)
        return out
    else:
        # leaf: obs는 array/스칼라, base_ndims는 int 또는 None
        x = jnp.asarray(obs)
        if isinstance(base_ndims, (int, jnp.integer)):
            spec_ndim = int(base_ndims)
        else:
            # 스펙이 없으면 합리적 기본값: 단일 샘플로 보고 배치 추가 필요
            # (이미 배치가 있으면 아래 ensure_batched_with_spec가 그대로 둠)
            # 이미지(H,W,C)=3, 벡터(D,)=1, 스칼라=0에 맞춰 동작
            spec_ndim = x.ndim if x.ndim in (1, 3) else max(0, x.ndim - 1)
        return ensure_batched_with_spec(x, spec_ndim)


from flax.core import FrozenDict, unfreeze

def _as_pydict(x):
    if isinstance(x, FrozenDict):
        x = unfreeze(x)
    if isinstance(x, dict):
        return {k: _as_pydict(v) for k, v in x.items()}
    return x

    
class FQLAgent(flax.struct.PyTreeNode):
    """Flow Q-learning (FQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the FQL critic loss."""
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

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = jnp.mean(qs, axis=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    # @jax.jit
    # def sample_actions(
    #     self,
    #     observations,
    #     seed=None,
    #     temperature=1.0,
    # ):
    #     """Sample actions from the one-step policy."""
    #     if seed is None:
    #         # jax.random.split는 seed 필요. 기본 처리
    #         seed = self.rng
    #     action_seed, noise_seed = jax.random.split(seed)

    #     # leaf = jax.tree_util.tree_leaves(observations)[0]
    #     # batch = leaf.shape[0]
    #     noises = jax.random.normal(
    #         action_seed,
    #         (
    #             *observations.shape[: -len(self.config['ob_dims'])],
    #             self.config['action_dim'],
    #         ),
    #     )
    #     actions = self.network.select('actor_onestep_flow')(observations, noises)
    #     actions = jnp.clip(actions, -1, 1)
    #     return actions

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        if seed is None:
            seed = self.rng
        action_seed, _ = jax.random.split(seed)

        base_ndims = _as_pydict(self.config['base_ndims'])
        observations = _as_pydict(observations)

        observations = _ensure_batched_tree(observations, base_ndims)

        first = jax.tree_util.tree_leaves(observations)[0]
        batch = first.shape[0]

        noises = jax.random.normal(action_seed, (batch, self.config['action_dim']))
        actions = self.network.select('actor_onestep_flow')(observations, noises)
        return jnp.clip(actions, -1, 1)


    @jax.jit
    def compute_flow_actions(self, observations, noises):
        base_ndims = _as_pydict(self.config['base_ndims'])
        observations = _as_pydict(observations)
        observations = _ensure_batched_tree(observations, base_ndims)

        first = jax.tree_util.tree_leaves(observations)[0]
        batch = first.shape[0]

        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((batch, 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_bc_flow')(observations, actions, t, is_encoded=False)
            actions = actions + vels / self.config['flow_steps']
        return jnp.clip(actions, -1, 1)



    @classmethod
    def create(
        cls,
        seed,
        ex_observations, 
        ex_actions,     
        config,
    ):
        """FQLAgent를 초기화한다. 
        포인트: 네트워크 init에 들어가는 모든 예시 입력의 배치 크기를 B=1로 통일한다.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # --- (1) 트리 리프(넘파이/제이엑스 배열)마다 [:1] 적용해서 B=1로 통일
        def _tree_first_batch(x):
            try:
                a = jnp.asarray(x)
                if a.ndim >= 1:
                    return a[:1]    # B=1
                return a
            except Exception:
                return x

        ex_observations_1 = jax.tree_util.tree_map(_tree_first_batch, ex_observations)
        ex_actions_1      = _tree_first_batch(ex_actions)
        ex_times_1        = ex_actions_1[..., :1]    # (1, 1) : flow 타임스텝 예시

        # --- (2) 모양 관련 메타 (샘플링 경로는 트리에서 B를 추론하므로 ob_dims 없어도 OK)
        if isinstance(ex_observations, dict):
            ob_dims = ()
        else:
            ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # --- (3) 인코더 구성 (픽셀 사용 여부에 따라)
        encoders = dict()
        encoder = None if config.get("use_pixels") == False else config["encoder"]
        if encoder is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic']             = encoder_module()
            encoders['actor_bc_flow']      = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # --- (4) 네트워크 정의
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

        # --- (5) init에 들어갈 "예시 입력"을 전부 B=1 버전으로 통일해서 등록
        network_info = dict(
            critic=(critic_def, (ex_observations_1, ex_actions_1)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations_1, ex_actions_1)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations_1, ex_actions_1, ex_times_1)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations_1, ex_actions_1)),
        )
        networks     = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        # --- (6) init & TrainState 생성
        network_def    = ModuleDict(networks)
        network_tx     = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network        = TrainState.create(network_def, network_params, tx=network_tx)

        # --- (7) 타겟 크리틱 초기화
        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        def _leaf_base_ndim(x):
            x = jnp.asarray(x)
            return int(x.ndim - 1)  # B=1이므로 배치 제외 랭크

        base_ndims = jax.tree_util.tree_map(_leaf_base_ndim, ex_observations_1)
        base_ndims = _as_pydict(base_ndims)            

        # config['ob_dims']     = ob_dims
        # config['action_dim']  = action_dim
        # config['base_ndims']  = base_ndims      
        # return cls(rng, network=network, config=flax.core.FrozenDict(**config))
            # --- (8) config 보정값 주입
        config['ob_dims']    = ob_dims
        config['action_dim'] = action_dim
        config['base_ndims']  = base_ndims   
        try:
            img_hw = tuple(ex_observations_1['pixels']['top'].shape[1:3])  # (H,W)
        except Exception:
            img_hw = (240, 320)  # fallback
        config['img_hw'] = img_hw

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))



    # @classmethod
    # def create(
    #     cls,
    #     seed,
    #     ex_observations,
    #     ex_actions,
    #     config,
    # ):
    #     """Create a new agent.

    #     Args:
    #         seed: Random seed.
    #         ex_observations: Example batch of observations.
    #         ex_actions: Example batch of actions.
    #         config: Configuration dictionary.
    #     """
    #     rng = jax.random.PRNGKey(seed)
    #     rng, init_rng = jax.random.split(rng, 2)

    #     ex_times = ex_actions[..., :1]
    #     # ob_dims = ex_observations.shape[1:]
    #     if isinstance(ex_observations, dict):
    #         ob_dims = ()
    #     else:
    #         ob_dims = ex_observations.shape[1:]
    #     action_dim = ex_actions.shape[-1]

    #     # Define encoders.
    #     encoders = dict()
    #     encoder = None if config["use_pixels"] == False else config["encoder"]
    #     if encoder is not None:
    #         encoder_module = encoder_modules[config['encoder']]
    #         encoders['critic'] = encoder_module()
    #         encoders['actor_bc_flow'] = encoder_module()
    #         encoders['actor_onestep_flow'] = encoder_module()

    #     # Define networks.
    #     critic_def = Value(
    #         hidden_dims=config['value_hidden_dims'],
    #         layer_norm=config['layer_norm'],
    #         num_ensembles=2,
    #         encoder=encoders.get('critic'),
    #     )
    #     actor_bc_flow_def = ActorVectorField(
    #         hidden_dims=config['actor_hidden_dims'],
    #         action_dim=action_dim,
    #         layer_norm=config['actor_layer_norm'],
    #         encoder=encoders.get('actor_bc_flow'),
    #     )
    #     actor_onestep_flow_def = ActorVectorField(
    #         hidden_dims=config['actor_hidden_dims'],
    #         action_dim=action_dim,
    #         layer_norm=config['actor_layer_norm'],
    #         encoder=encoders.get('actor_onestep_flow'),
    #     )

    #     network_info = dict(
    #         critic=(critic_def, (ex_observations, ex_actions)),
    #         target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
    #         actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
    #         actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_actions)),
    #     )
    #     # if encoders.get('actor_bc_flow') is not None:
    #     #     # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
    #     #     network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
    #     networks = {k: v[0] for k, v in network_info.items()}
    #     network_args = {k: v[1] for k, v in network_info.items()}

    #     network_def = ModuleDict(networks)
    #     network_tx = optax.adam(learning_rate=config['lr'])
    #     network_params = network_def.init(init_rng, **network_args)['params']
    #     network = TrainState.create(network_def, network_params, tx=network_tx)

    #     params = network.params
    #     params['modules_target_critic'] = params['modules_critic']

    #     config['ob_dims'] = ob_dims
    #     config['action_dim'] = action_dim
    #     return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fql',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),

            # Optim
            lr=3e-4,
            batch_size=64,

            # Networks
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,

            # RL
            discount=0.99,
            tau=0.005,
            q_agg='mean',        # or 'min'
            alpha=10.0,
            flow_steps=10,
            normalize_q_loss=False,

            # ---- Encoder 선택 ----
            # 이미지+agent_pos를 기본으로 쓸 거면 'aloha_mm'
            # state-only면 None으로만 바꾸면 됩니다.
            use_pixels = True,
            encoder='aloha_mm',

            encoder_kwargs=dict(
                impala_num_blocks=1,
                impala_stack_sizes=(16, 32, 32),
                impala_width=1,
                impala_layer_norm=False,
                impala_mlp_hidden=(512,),
                proprio_hidden=(256,),
                proprio_layer_norm=False,
            ),

            obs_keys=dict(
                pixels_top=('pixels', 'top'),
                agent_pos=('agent_pos',),
            ),
        )
    )
    return config


# def get_config():
#     config = ml_collections.ConfigDict(
#         dict(
#             agent_name='fql',  # Agent name.
#             ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
#             action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
#             lr=3e-4,  # Learning rate.
#             batch_size=256,  # Batch size.
#             actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
#             value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
#             layer_norm=True,  # Whether to use layer normalization.
#             actor_layer_norm=False,  # Whether to use layer normalization for the actor.
#             discount=0.99,  # Discount factor.
#             tau=0.005,  # Target network update rate.
#             q_agg='mean',  # Aggregation method for target Q values.
#             alpha=10.0,  # BC coefficient (need to be tuned for each environment).
#             flow_steps=10,  # Number of flow steps.
#             normalize_q_loss=False,  # Whether to normalize the Q loss.
#             encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
#         )
#     )
#     return config
