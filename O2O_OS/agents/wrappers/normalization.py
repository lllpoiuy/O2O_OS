from typing import Dict

import numpy as np
from flax.training import checkpoints

# from O2O_OS.agents.base_agent import AgentWrapper, BaseAgent
from agents.wrappers.utils import RunningMeanStd

import flax
import os
import pickle
from typing import Any, Dict
import jax
import jax.numpy as jnp
from functools import partial
from jax.typing import ArrayLike
from flax import struct
from utils.flax_utils import nonpytree_field

class NormalizedAgent(flax.struct.PyTreeNode):
    """
    A PyTreeNode wrapper that normalizes observations before forwarding to
    an underlying O2O_OS_Agent. Supports full jit on update without custom hashing.
    """
    agent: Any
    obs_rms: Any = nonpytree_field()  # running mean/var not traced by JAX
    epsilon: float = nonpytree_field()
    config: Any = nonpytree_field()
    
    def __hash__(self):
        # Use object identity hash as a fallback to avoid hashing JAX arrays
        return id(self)

    @classmethod
    def create(cls, agent: Any, obs_shape: tuple, epsilon: float = 1e-8):
        """
        Factory method to initialize a new NormalizedAgent with default RunningMeanStd.
        """
        # Initialize RunningMeanStd with zeros/ones
        rms = RunningMeanStd(shape=obs_shape)
        return cls(agent=agent, obs_rms=rms, epsilon=epsilon, config=agent.config,)

    def _normalize(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Normalize observations: (obs - mean) / sqrt(var + epsilon)."""
        mean = jnp.asarray(self.obs_rms.mean)
        var = jnp.asarray(self.obs_rms.var)
        return (obs - mean) / jnp.sqrt(var + self.epsilon)

    @staticmethod
    def _update(wrapper, batch):
        """
        Static update: normalize batch, apply agent update, return new wrapper and info.
        """
        batch = batch.copy()  # avoid in-place mutation
        # Normalize observations
        obs = jnp.asarray(batch['observations'])
        next_obs = jnp.asarray(batch['next_observations'])
        batch['observations'] = wrapper._normalize(obs)
        batch['next_observations'] = wrapper._normalize(next_obs)

        # Delegate to underlying agent
        agent = wrapper.agent
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(params):
            return agent.total_loss(batch, params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        new_agent = agent.replace(network=new_network, rng=new_rng)

        # Return updated wrapper and info
        return wrapper.replace(agent=new_agent), info

    # @partial(jax.jit, static_argnums=0)
    @jax.jit
    def update(self, batch):
        """Jitted update: returns (new_wrapper, info)."""
        return self._update(self, batch)

    # @partial(jax.jit, static_argnums=0)
    @jax.jit
    def batch_update(self, batch):
        """
        Jitted batch update using lax.scan. Returns (new_wrapper, averaged_info).
        """
        wrapper, infos = jax.lax.scan(self._update, self, batch)
        mean_info = jax.tree_util.tree_map(lambda x: x.mean(), infos)
        return wrapper, mean_info

    # @partial(jax.jit, static_argnums=0)
    # @jax.jit
    # @partial(jax.jit, static_argnames=['training'])
    def sample_actions(
        self,
        observations: jnp.ndarray,
        rng: jax.random.PRNGKey,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Normalize and optionally update stats, then sample actions.
        """
        if training:
            # Update running stats on Python side
            self.obs_rms.update(observations)

        normalized = self._normalize(observations)
        return self.agent.sample_actions(normalized, rng=rng)