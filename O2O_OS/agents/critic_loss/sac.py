import flax
import jax
import jax.numpy as jnp


def critic_loss(
        agent : flax.struct.PyTreeNode,
        batch : dict,
        grad_params : dict,
        batch_actions : jnp.ndarray,
        rng : jax.random.PRNGKey,
    ) -> tuple:
    """
    Compute the SAC critic loss.
    Example config:
    "critic_loss": {
        "type": "sac",
        "q_agg": "mean"
    }
    """

    assert agent.network.select('critic') is not None
    assert agent.network.select('target_critic') is not None
    critic_loss_config = agent.config["critic_loss"]

    rng, sample_rng = jax.random.split(rng)

    next_actions = agent.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

    next_qs = agent.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)

    if critic_loss_config['q_agg'] == 'min':
        next_q = next_qs.min(axis=0)
    else:
        next_q = next_qs.mean(axis=0)

    target_q = batch['rewards'][..., -1] + (agent.config['discount'] ** agent.config["horizon_length"]) * batch['masks'][..., -1] * next_q
    
    q = agent.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
    critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

    return critic_loss, {
        'critic_loss': critic_loss,
        'q_mean': q.mean(),
        'q_max': q.max(),
        'q_min': q.min(),
    }