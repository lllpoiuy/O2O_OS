import flax.linen as nn
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from networks.simbaV2_networks.simbaV2_layer import (
    HyperCategoricalValue,
    HyperEmbedder,
    HyperLERPBlock,
    HyperNormalTanhPolicy,
)


tfd = tfp.distributions
tfb = tfp.bijectors


class SimbaV2Actor(nn.Module):
    num_blocks: int
    hidden_dim: int
    action_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float

    def setup(self):
        self.embedder = HyperEmbedder(  # [NOTE] Shift + l_2 norm & Linear + Scaler, SimBaV2 / trick-1.1 & trick-1.2
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential(  # [NOTE] MLP + l_2 norm & LERP + l_2 norm, SimBaV2 / trick-2.1 & trick-2.2
            [
                HyperLERPBlock(
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.predictor = HyperNormalTanhPolicy(
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        temperature: float = 1.0,
    ) -> tfd.Distribution:
        x = observations
        y = self.embedder(x)
        z = self.encoder(y)
        dist, info = self.predictor(z, temperature)

        # return dist, info
        return dist


class SimbaV2Critic(nn.Module):
    num_blocks: int
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float
    num_bins: int
    min_v: float
    max_v: float

    def setup(self):
        self.embedder = HyperEmbedder(  # [NOTE] Shift + l_2 norm & Linear + Scaler, SimBaV2 / trick-1.1 & trick-1.2
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            c_shift=self.c_shift,
        )
        self.encoder = nn.Sequential(  # [NOTE] MLP + l_2 norm & LERP + l_2 norm, SimBaV2 / trick-2.1 & trick-2.2
            [
                HyperLERPBlock(
                    hidden_dim=self.hidden_dim,
                    scaler_init=self.scaler_init,
                    scaler_scale=self.scaler_scale,
                    alpha_init=self.alpha_init,
                    alpha_scale=self.alpha_scale,
                )
                for _ in range(self.num_blocks)
            ]
        )

        self.predictor = HyperCategoricalValue(  # [NOTE] Distributional Critic, SimBaV2 / trick-3.1
            hidden_dim=self.hidden_dim,
            num_bins=self.num_bins,
            min_v=self.min_v,
            max_v=self.max_v,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        # print(f"SimbaV2Critic: observations.shape={observations.shape}, actions.shape={actions.shape}")
        x = jnp.concatenate((observations, actions), axis=-1)
        y = self.embedder(x)
        z = self.encoder(y)
        q, info = self.predictor(z)
        # return q, info
        return q


class SimbaV2DoubleCritic(nn.Module):
    """
    Vectorized Double-Q for Clipped Double Q-learning.
    https://arxiv.org/pdf/1802.09477v3
    """

    num_blocks: int
    hidden_dim: int
    scaler_init: float
    scaler_scale: float
    alpha_init: float
    alpha_scale: float
    c_shift: float
    num_bins: int
    min_v: float
    max_v: float

    num_qs: int = 2

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        VmapCritic = nn.vmap(
            SimbaV2Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=0,
            axis_size=self.num_qs,
        )

        qs, infos = VmapCritic(
            num_blocks=self.num_blocks,
            hidden_dim=self.hidden_dim,
            scaler_init=self.scaler_init,
            scaler_scale=self.scaler_scale,
            alpha_init=self.alpha_init,
            alpha_scale=self.alpha_scale,
            c_shift=self.c_shift,
            num_bins=self.num_bins,
            min_v=self.min_v,
            max_v=self.max_v,
        )(observations, actions)

        # return qs, infos
        return qs
