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

@struct.dataclass
class NormalizedAgent:
    """
    A Flax PyTreeNode wrapper that normalizes observations before
    forwarding to an underlying O2O_OS_Agent. Supports full jit on update.
    """
    agent: Any
    obs_rms: Any = struct.field(pytree_node=False)  # running mean/var not traced by JAX
    epsilon: float = 1e-8
    
    def __hash__(self):
        """
        Custom hash to avoid hashing JAX arrays inside agent.
        Use object id to ensure hashability without touching array contents.
        """
        return hash(id(self))

    @classmethod
    def create(cls, agent: Any, obs_shape: tuple, epsilon: float = 1e-8):
        """
        Factory method to initialize a new NormalizedAgent with default RunningMeanStd.
        """
        from agents.wrappers.normalization import RunningMeanStd
        rms = RunningMeanStd(shape=obs_shape)
        return cls(agent=agent, obs_rms=rms, epsilon=epsilon)

    def _normalize(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Normalize observations: (obs - mean) / sqrt(var + epsilon)."""
        mean = jnp.asarray(self.obs_rms.mean)
        var = jnp.asarray(self.obs_rms.var)
        return (obs - mean) / jnp.sqrt(var + self.epsilon)

    @staticmethod
    def _update(wrapper, batch):
        """
        Static update function: normalizes batch, applies agent update,
        and returns new wrapper plus info dict.
        """
        # Copy batch to avoid in-place mutation
        batch = batch.copy()
        # Convert observations to JAX arrays and normalize
        obs = jnp.asarray(batch["observations"])
        next_obs = jnp.asarray(batch["next_observations"])
        batch["observations"] = wrapper._normalize(obs)
        batch["next_observations"] = wrapper._normalize(next_obs)

        # Delegate update to underlying agent
        agent = wrapper.agent
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        new_agent = agent.replace(network=new_network, rng=new_rng)

        # Return updated wrapper and info
        return wrapper.replace(agent=new_agent), info

    @partial(jax.jit, static_argnums=0)
    def update(self, batch):
        """Full jitted update: returns (new_wrapper, info)."""
        return self._update(self, batch)

    @partial(jax.jit, static_argnums=0)
    def sample_actions(
        self,
        observations: jnp.ndarray,
        rng: jax.random.PRNGKey,
        training: bool = False,
    ) -> jnp.ndarray:
        """
        Normalize and update stats (if training), then sample actions.
        """
        # Update running stats on Python side only
        if training:
            self.obs_rms.update(observations)

        normalized = self._normalize(observations)
        return self.agent.sample_actions(normalized, rng=rng)

    @partial(jax.jit, static_argnums=0)
    def batch_update(self, batch):
        """
        Jitted batch update using lax.scan over _update.
        Returns (new_wrapper, averaged_info).
        """
        # Use scan to apply _update repeatedly over batch entries
        wrapper, infos = jax.lax.scan(self._update, self, batch)
        mean_info = jax.tree_util.tree_map(lambda x: x.mean(), infos)
        return wrapper, mean_info