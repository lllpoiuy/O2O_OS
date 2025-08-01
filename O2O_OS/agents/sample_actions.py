import flax
import jax
import jax.numpy as jnp
from agents.wrappers.normalization import NormalizedAgent


def sample_dist(
        agent : flax.struct.PyTreeNode,
        observations : dict,
        rng : dict
    ) -> tuple:
    """
    Sample actions from the Gaussian actor distribution.
    Example config:
    "sample_actions":{
        "type": "gaussian"
    }
    """

    dist = agent.network.select('actor')(observations)
    actions = dist.sample(seed=rng)
    actions = jnp.clip(actions, -1, 1)
    return actions



@jax.jit
def compute_flow_actions(
    agent : flax.struct.PyTreeNode,
    observations,
    noises,
):
    """Compute actions from the BC flow model using the Euler method."""
    
    if isinstance(agent, NormalizedAgent):
        agent = agent.agent
    if agent.config['encoder'] is not None:
        observations = agent.network.select('actor_bc_flow_encoder')(observations)
    actions = noises
    # Euler method.
    for i in range(agent.config["create_network"]['flow_steps']):
        t = jnp.full((*observations.shape[:-1], 1), i / agent.config["create_network"]['flow_steps'])
        vels = agent.network.select('actor_bc_flow')(observations, actions, t, is_encoded=True)
        actions = actions + vels / agent.config["create_network"]['flow_steps']
    actions = jnp.clip(actions, -1, 1)
    return actions



def sample_best_of_n(
        agent : flax.struct.PyTreeNode,
        observations : dict,
        rng : dict
    ) -> tuple:

    """
    Sample the best action from n samples.
    Example config:
    "sample_actions":{
        "type": "best_of_n",
        "actor_num_samples": 32,
        "soft_max": true
    }
    """
    assert agent.network.select('actor_bc_flow') is not None
    assert agent.network.select('critic') is not None
    assert agent.config['horizon_length'] is not None
    assert agent.config['action_dim'] is not None
    assert agent.config['sample_actions']['actor_num_samples'] is not None

    action_dim = agent.config['action_dim'] * agent.config['horizon_length']
    noises = jax.random.normal(
        rng,
        (
            *observations.shape[: -len(agent.config['ob_dims'])],
            agent.config["sample_actions"]["actor_num_samples"], action_dim
        ),
    )
    observations = jnp.repeat(observations[..., None, :], agent.config["sample_actions"]["actor_num_samples"], axis=-2)
    actions = compute_flow_actions(agent, observations, noises)
    actions = jnp.clip(actions, -1, 1)
    if agent.config["critic_loss"]["q_agg"] == "mean":
        q = agent.network.select("critic")(observations, actions).mean(axis=0)
    else:
        q = agent.network.select("critic")(observations, actions).min(axis=0)

    if agent.config['sample_actions'].get('soft_max', False):
        q = jax.nn.softmax(q, axis=-1)
        indices = jax.random.categorical(rng, q, axis=-1)
    else:
        indices = jnp.argmax(q, axis=-1)

    bshape = indices.shape
    indices = indices.reshape(-1)
    bsize = len(indices)
    actions = jnp.reshape(actions, (-1, agent.config["sample_actions"]["actor_num_samples"], action_dim))[jnp.arange(bsize), indices, :].reshape(bshape + (action_dim,))

    return actions



def sample_distill_ddpg(
        agent : flax.struct.PyTreeNode,
        observations : dict,
        rng : dict,
    ) -> tuple:
    """
    Sample actions using the distill DDPG method with best-of-N sampling.
    Example config:
    "sample_actions":{
        "type": "distill_ddpg",
        "actor_num_samples": 32,
        "soft_max": true,
        "imitation_boostrapped": true
    }
    """
    assert agent.network.select('actor_onestep_flow') is not None
    assert agent.network.select('critic') is not None
    assert agent.config['sample_actions'].get('type') is not None
    assert agent.config['sample_actions'].get('actor_num_samples') is not None
    assert agent.config['sample_actions'].get('soft_max') is not None
    assert agent.config['sample_actions'].get('imitation_boostrapped') is not None
    
    action_dim = agent.config['action_dim'] * agent.config['horizon_length']
    num_samples = agent.config['sample_actions']['actor_num_samples']
    
    noises = jax.random.normal(
        rng,
        (
            *observations.shape[: -len(agent.config['ob_dims'])],
            num_samples, action_dim
        ),
    )
    
    observations_repeated = jnp.repeat(observations[..., None, :], num_samples, axis=-2)
    
    actions = agent.network.select('actor_onestep_flow')(observations_repeated, noises)
    actions = jnp.clip(actions, -1, 1)
    
    if agent.config["critic_loss"]["q_agg"] == "mean":
        q = agent.network.select("critic")(observations_repeated, actions).mean(axis=0)
    else:
        q = agent.network.select("critic")(observations_repeated, actions).min(axis=0)
    
    if agent.config['sample_actions'].get('soft_max', False):
        q = jax.nn.softmax(q, axis=-1)
        indices = jax.random.categorical(rng, q, axis=-1)
    else:
        indices = jnp.argmax(q, axis=-1)
    
    bshape = indices.shape
    indices = indices.reshape(-1)
    bsize = len(indices)
    actions = jnp.reshape(actions, (-1, num_samples, action_dim))[jnp.arange(bsize), indices, :].reshape(bshape + (action_dim,))
    q_actions = jnp.reshape(q, (-1, num_samples))[jnp.arange(bsize), indices].reshape(bshape)

    if agent.config['sample_actions']['imitation_boostrapped']:
        imitation_noise = jax.random.normal(
            rng,
            (
                *observations.shape[: -len(agent.config['ob_dims'])],
                1, action_dim
            ),
        )
        imitation_observations = jnp.repeat(observations[..., None, :], 1, axis=-2)
        imitation_actions = compute_flow_actions(agent, imitation_observations, imitation_noise)
        imitation_actions = jnp.clip(imitation_actions, -1, 1)

        q_imitation = agent.network.select("critic")(imitation_observations, imitation_actions)
        if agent.config["critic_loss"]["q_agg"] == "mean":
            q_imitation = q_imitation.mean(axis=0)
        else:
            q_imitation = q_imitation.min(axis=0)

        # print("---q_imitation.shape:", q_imitation.shape)
        # print("---q_actions.shape:", q_actions.shape)
        # print("---imitation_actions.shape:", imitation_actions.shape)
        # print("---actions.shape:", actions.shape)
        
        q_imitation = jnp.squeeze(q_imitation, axis=-1)
        imitation_actions = jnp.squeeze(imitation_actions, axis=-2)

        q_actions = jnp.where(
            jnp.expand_dims(q_imitation, axis=-1) > jnp.expand_dims(q_actions, axis=-1),
            imitation_actions, actions
        )
    
    return actions