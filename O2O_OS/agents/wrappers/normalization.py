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


class NormalizedAgent:
    """
    A stateful wrapper that always normalizes observations before
    forwarding to an underlying O2O_OS_Agent.

    It maintains running mean and variance in `obs_rms`.
    """
    def __init__(self, agent: Any, obs_shape: tuple, epsilon: float = 1e-8):
        self.agent = agent
        self.obs_rms = RunningMeanStd(shape=obs_shape)
        self.epsilon = epsilon

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations with (obs - mean) / sqrt(var + epsilon)."""
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

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
        norm_jax = jnp.asarray(normalized)
        actions = self.agent.sample_actions(norm_jax, rng=rng)
        return actions

    @staticmethod
    def _update(agent, batch):
        batch = batch.copy()
        obs = np.array(batch["observations"])
        next_obs = np.array(batch["next_observations"])
        batch["observations"]      = jnp.asarray(agent._normalize(obs))
        batch["next_observations"] = jnp.asarray(agent._normalize(next_obs))
        
        new_rng, rng = jax.random.split(agent.rng)
        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)
        
        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        
        return agent.replace(network=new_network, rng=new_rng), info
    
    @jax.jit
    def update(self, batch):
        return self._update(self.agent, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        # update_size = batch["observations"].shape[0]
        agent, infos = jax.lax.scan(self._update, self.agent, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)
    