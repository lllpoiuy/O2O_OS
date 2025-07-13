import flax
import jax
import jax.numpy as jnp


def sac_bc(
        agent : flax.struct.PyTreeNode,
        batch : dict,
        grad_params : dict,
        rng : jax.random.PRNGKey,
    ) -> tuple:
    """
    Compute the SAC actor loss.
    For Gaussian policy only, actor network must return a distribution.
    Example config:
    "actor_loss": {
        "type": "sac_bc",
        "target_entropy": null,
        "target_entropy_multiplier": 1.0,
        "alpha": 1.0,
        "bc_alpha": 0.0,
        "init_temp": 1.0
    }
    """

    assert agent.network.select('actor') is not None
    assert agent.network.select('critic') is not None
    assert agent.network.select('alpha') is not None
    actor_loss_config = agent.config["actor_loss"]
    assert actor_loss_config["target_entropy"] is not None
    assert actor_loss_config["bc_alpha"] is not None

    if agent.config["action_chunking"]:
        batch_actions = jnp.reshape(batch["actions"], (batch["actions"].shape[0], -1))
    else:
        batch_actions = batch["actions"][..., 0, :]

    dist = agent.network.select('actor')(batch['observations'], params=grad_params)
    # assert isinstance(dist, jax.distributions.Distribution), "Actor network must return a distribution."
    actions = dist.sample(seed=rng)
    log_probs = dist.log_prob(actions)

    # Actor loss.
    qs = agent.network.select('critic')(batch['observations'], actions)
    q = jnp.mean(qs, axis=0) if agent.config["critic_loss"]["q_agg"] == "mean" else jnp.min(qs, axis=0)

    actor_loss = (log_probs * agent.network.select('alpha')() - q).mean()

    # Entropy loss.
    alpha = agent.network.select('alpha')(params=grad_params)
    entropy = -jax.lax.stop_gradient(log_probs).mean()
    alpha_loss = (alpha * (entropy - actor_loss_config['target_entropy'])).mean()

    # BC loss.
    bc_loss = -dist.log_prob(jnp.clip(batch_actions, -1 + 1e-5, 1 - 1e-5)).mean() * actor_loss_config["bc_alpha"]

    total_loss = actor_loss + alpha_loss + bc_loss

    return total_loss, {
        'total_loss': total_loss,
        'actor_loss': actor_loss,
        'alpha_loss': alpha_loss,
        'bc_loss': bc_loss,
        'alpha': alpha,
        'entropy': -log_probs.mean(),
        'q': q.mean(),
    }

