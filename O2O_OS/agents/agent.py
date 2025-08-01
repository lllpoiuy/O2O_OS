from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections

from utils.flax_utils import nonpytree_field

import agents


class O2O_OS_Agent(flax.struct.PyTreeNode):
    """Soft actor-critic (SAC) agent.

    This agent can also be used for reinforcement learning with prior data (RLPD).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the SAC critic loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        
        if self.config["critic_loss"]["type"] == "sac":
            return agents.critic_loss.critic_loss(
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
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        
        if self.config["actor_loss"]["type"] == "sac_bc":
            return agents.actor_loss.sac_bc(
                agent=self,
                batch=batch,
                grad_params=grad_params,
                batch_actions = batch_actions,
                rng=rng
            )
        elif self.config["actor_loss"]["type"] == "flow_bc":
            return agents.actor_loss.bc_flow(
                agent=self,
                batch=batch,
                grad_params=grad_params,
                batch_actions=batch_actions,
                rng=rng
            )
        else:
            raise ValueError(f"Unknown actor loss type: {self.config['actor_loss']['type']}")
    
    def imitation_loss(self, batch, grad_params, rng):
        """Compute the imitation loss."""
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
        
        return agents.actor_loss.imitation_loss(
            agent=self,
            batch=batch,
            grad_params=grad_params,
            batch_actions=batch_actions,
            rng=rng
        )
        
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
    
    @staticmethod
    def _warmup_update(self, batch):
        """Warmup update for the agent."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            critic_loss, critic_info = self.critic_loss(batch, grad_params, rng=rng)
            info = {}
            for k, v in critic_info.items():
                info[f'critic/{k}'] = v
            return critic_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info
    
    @jax.jit
    def warmup_update(self, batch):
        return self._warmup_update(self, batch)
    
    @jax.jit
    def batch_warmup_update(self, batch):
        agent, infos = jax.lax.scan(self._warmup_update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    
    @jax.jit
    def imitation_update(self, batch):
        """Update the agent using imitation learning."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.imitation_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info
    
    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
        training: bool = False,
    ):
        """Sample actions from the actor."""
        if self.config["sample_actions"]["type"] == "gaussian":
            return agents.sample_actions.sample_dist(
                agent=self,
                observations=observations,
                rng=rng
            )
        elif self.config["sample_actions"]["type"] == "best_of_n":
            return agents.sample_actions.sample_best_of_n(
                agent=self,
                observations=observations,
                rng=rng
            )
        elif self.config["sample_actions"]["type"] == "distill_ddpg":
            return agents.sample_actions.sample_distill_ddpg(
                agent=self,
                observations=observations,
                rng=rng
            )
        else:
            raise ValueError(f"Unknown sample actions type: {self.config['sample_actions']['type']}")

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

        if config["create_network"]["type"] == "flow":
            network = agents.create_network.create_flow_network(
                config=config,
                init_rng=init_rng,
                ex_observations=ex_observations,
                ex_actions=ex_actions,
            )
        elif config["create_network"]["type"] == "normal":
            network = agents.create_network.create_normal_network(
                config=config,
                init_rng=init_rng,
                ex_observations=ex_observations,
                ex_actions=ex_actions,
            )
        else:
            raise ValueError(f"Unknown create network type: {config['create_network']['type']}")

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


import ml_collections
import json


def get_config():
    print("getting config...")
    with open('./agents/configs/config.json', 'r') as f:
        config_dict = json.load(f)
    print("Loaded config:", config_dict)
    config = ml_collections.ConfigDict(config_dict)
    return config
