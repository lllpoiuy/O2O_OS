import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import linen as nn

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from rlpd_networks import Ensemble, StateActionValue, MLP
from rlpd_distributions import TanhNormal

from functools import partial

import agents

class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


class O2O_OS_Agent(flax.struct.PyTreeNode):
    """Soft actor-critic (SAC) agent.

    This agent can also be used for reinforcement learning with prior data (RLPD).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the SAC critic loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        if self.config["critic_loss"]["type"] == "sac":
            return agents.critic_loss.sac.critic_loss(
                agent=self,
                batch=batch,
                grad_params=grad_params,
                batch_actions=batch_actions,
                rng=rng
            )
        else:
            raise ValueError(f"Unknown critic loss type: {self.config['critic_loss']['type']}")
    
    def actor_loss(self, batch, grad_params, rng):
        """Compute the SAC actor loss."""
        if self.config["action_chunking"]:
            batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        else:
            batch_actions = batch["actions"][..., 0, :]
        
        if self.config["actor_loss"]["type"] == "sac_bc":
            return agents.actor_loss.sac_bc.actor_loss(
                agent=self,
                batch=batch,
                grad_params=grad_params,
                rng=rng
            )
        else:
            raise ValueError(f"Unknown actor loss type: {self.config['actor_loss']['type']}")
        
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
    
    @staticmethod
    def _update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info
    
    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations)
        actions = dist.sample(seed=rng)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        if config["actor_loss"]['target_entropy'] is None:
            config["actor_loss"]['target_entropy'] = -config["actor_loss"]['target_entropy_multiplier'] * full_action_dim

        # Define networks
        critic_base_cls = partial(
            MLP,
            hidden_dims=config['value_hidden_dims'],
            activate_final=True,
            use_layer_norm=config["critic_layer_norm"],
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=config["num_qs"])


        actor_base_cls = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
        actor_def = TanhNormal(actor_base_cls, full_action_dim)

        # Define the dual alpha variable.
        alpha_def = Temperature(config["actor_loss"]["init_temp"])

        network_info = dict(
            critic=(critic_def, (ex_observations, full_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
            actor=(actor_def, (ex_observations,)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


import ml_collections
import json

def get_config():
    with open('./agents/config.json', 'r') as f:
        config_dict = json.load(f)
    config = ml_collections.ConfigDict(config_dict)
    return config
