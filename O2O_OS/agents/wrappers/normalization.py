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


class NormalizedAgent:
    """
    A stateful wrapper that always normalizes observations before
    forwarding to an underlying O2O_OS_Agent.

    It maintains running mean and variance in `obs_rms`.
    """
    def __init__(self, agent: Any, obs_shape: tuple, epsilon: float = 1e-8, obs_rms=None):
        self.agent = agent
        self.obs_rms = obs_rms if obs_rms is not None else RunningMeanStd(shape=obs_shape)
        self.epsilon = epsilon
    
    def _normalize(self, obs: ArrayLike) -> jax.Array:
        """Normalize observations with (obs - mean) / sqrt(var + epsilon)."""
        mean = jnp.asarray(self.obs_rms.mean)
        var = jnp.asarray(self.obs_rms.var)
        return (obs - mean) / jnp.sqrt(var + self.epsilon)
    
    def replace(self, *, agent=None, obs_rms=None):
        """Return a new NormalizedAgent with updated fields."""
        return NormalizedAgent(
            agent=agent if agent is not None else self.agent,
            obs_shape=self.obs_rms.mean.shape,
            epsilon=self.epsilon,
            obs_rms=obs_rms if obs_rms is not None else self.obs_rms,
        )

    def sample_actions(
        self,
        observations: np.ndarray,
        rng: jax.random.PRNGKey,
        training: bool = False,
    ) -> np.ndarray:
        """
        Normalize and update stats (if training), then call agent.sample_actions.

        Args:
            observations: raw numpy array of observations.
            rng: JAX random key.
            training: whether to update running statistics.

        Returns:
            actions produced by the underlying agent.
        """
        if training:
            self.obs_rms.update(observations)

        normalized = self._normalize(observations)
        actions = self.agent.sample_actions(normalized, rng=rng)
        return actions

    @staticmethod
    def _update(wrapper, batch):
        batch = batch.copy()
        obs = jnp.asarray(batch["observations"])
        next_obs = jnp.asarray(batch["next_observations"])
        batch["observations"] = wrapper._normalize(obs)
        batch["next_observations"] = wrapper._normalize(next_obs)
        
        agent = wrapper.agent
        
        new_rng, rng = jax.random.split(agent.rng)
        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)
        
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        agent = agent.replace(network=new_network, rng=new_rng)
        
        return wrapper.replace(agent=agent), info
    
    @partial(jax.jit, static_argnums=0)
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self.agent, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    