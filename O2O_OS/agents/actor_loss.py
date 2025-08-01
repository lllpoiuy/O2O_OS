import flax
import jax
import jax.numpy as jnp

import agents
import agents.sample_actions

def sac_bc(
        agent : flax.struct.PyTreeNode,
        batch : dict,
        grad_params : dict,
        rng : jax.random.PRNGKey,
        batch_actions : jnp.ndarray,
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


def bc_flow(
        agent : flax.struct.PyTreeNode,
        batch : dict,
        grad_params : dict,
        rng : jax.random.PRNGKey,
        batch_actions : jnp.ndarray,
    ) -> tuple:
    """
    Compute the Flow BC actor loss.
    Example config:
    "actor_loss": {
        "type": "flow_bc",
        "alpha": 1.0
    }
    """
    assert agent.network.select('actor_bc_flow') is not None
    assert agent.network.select('critic') is not None
    actor_loss_config = agent.config["actor_loss"]

    batch_size, action_dim = batch_actions.shape
    rng, x_rng, t_rng, n_rng = jax.random.split(rng, 4)

    # BC flow loss.
    x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
    x_1 = batch_actions
    t = jax.random.uniform(t_rng, (batch_size, 1))
    x_t = (1 - t) * x_0 + t * x_1
    vel = x_1 - x_0

    pred = agent.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

    bc_flow_loss = jnp.reshape(
            (pred - vel) ** 2, 
            (batch_size, agent.config["horizon_length"], agent.config["action_dim"]) 
        ) * batch["valid"][..., None]
    
    q_batch = agent.network.select('critic')(batch['observations'], batch_actions)

    obac_rate = None

    if actor_loss_config.get("OBAC", None) is not None and actor_loss_config["OBAC"] > 0:

        print("OBAC enabled, computing OBAC loss.")

        now_action = agent.sample_actions(
            batch['observations'],
            rng=n_rng,
        )
        now_action = jnp.clip(now_action, -1 + 1e-5, 1 - 1e-5)
        q_now = agent.network.select('critic')(batch['observations'], now_action)

        if agent.config['critic_loss']['q_agg'] == 'mean':
            q_batch = jnp.mean(q_batch, axis=0)
            q_now = jnp.mean(q_now, axis=0)
        else:
            q_batch = jnp.min(q_batch, axis=0)
            q_now = jnp.min(q_now, axis=0)

        mask = (q_batch > q_now)[..., None, None]
        bc_flow_loss = bc_flow_loss + (bc_flow_loss * mask) * actor_loss_config["OBAC"]
        obac_rate = jnp.mean(mask)

    actor_loss = jnp.mean(bc_flow_loss)

    if agent.config["sample_actions"]["type"] == "distill_ddpg":
        assert agent.network.select('actor_onestep_flow') is not None
        assert actor_loss_config.get("alpha", None) is not None

        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = agents.sample_actions.compute_flow_actions(
            agent,
            batch['observations'],
            noises=noises,
        )
        actor_actions = agent.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)
        
        # Q loss.
        actor_actions = jnp.clip(actor_actions, -1, 1)

        qs = agent.network.select(f'critic')(batch['observations'], actor_actions)
        q = jnp.mean(qs, axis=0)
        q_loss = -q.mean()

        actor_loss += distill_loss * actor_loss_config["alpha"] + q_loss
    
    metrics = {
        'total_loss': actor_loss,
        'actor_loss': actor_loss,
        'bc_flow_loss': jnp.mean(bc_flow_loss)
    }
    if obac_rate is not None:
        metrics['obac_rate'] = obac_rate


    return actor_loss, metrics

def imitation_loss(
        agent : flax.struct.PyTreeNode,
        batch : dict,
        grad_params : dict,
        batch_actions : jnp.ndarray,
        rng : jax.random.PRNGKey,
    ) -> tuple:

    metrics = {}
    total_loss = 0.0
    
    if agent.config["sample_actions"]["type"] == "gaussian":

        dist = agent.network.select('actor')(batch['observations'], params=grad_params)
        bc_loss = -dist.log_prob(jnp.clip(batch_actions, -1 + 1e-5, 1 - 1e-5)).mean()
        total_loss += bc_loss
        metrics['bc_loss'] = bc_loss

    elif agent.config["sample_actions"]["type"] == "best_of_n" or agent.config["sample_actions"]["type"] == "distill_ddpg":

        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = agent.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)

        bc_flow_loss = jnp.reshape(
                (pred - vel) ** 2, 
                (batch_size, agent.config["horizon_length"], agent.config["action_dim"]) 
            ) * batch["valid"][..., None]
        
        bc_flow_loss = jnp.mean(bc_flow_loss)
        total_loss += bc_flow_loss
        metrics['bc_flow_loss'] = bc_flow_loss

        if agent.config["sample_actions"]["type"] == "distill_ddpg":
            assert agent.network.select('actor_onestep_flow') is not None
            assert agent.config["actor_loss"].get("alpha", None) is not None

            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))
            target_flow_actions = agents.sample_actions.compute_flow_actions(
                agent,
                batch['observations'],
                noises=noises,
            )
            actor_actions = agent.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
            distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

            total_loss += distill_loss * agent.config["actor_loss"]["alpha"]
            metrics['distill_loss'] = distill_loss
    
    else:
        raise ValueError(f"Unknown sample actions type: {agent.config['sample_actions']['type']}")
    
    return total_loss, metrics
