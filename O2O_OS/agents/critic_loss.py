import flax
import jax
import jax.numpy as jnp

# reshape issue: batch['valid]?

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
        "a_next": "BFN / base",
        "q_agg": "mean",
        "td_target" : "single / double / mean / min",
        "cql": {
            "cql_n_actions": 10,
            "cql_temperature": 1,
            "cql_min_q_weight": 0.005,
            "cql_min_q_weight_online": 0.0,
            "cql_target_action_gap": 1.0,
            "cql_log_alpha_prime": 1.0
        }
    }
    """

    assert agent.network.select('critic') is not None
    assert agent.network.select('target_critic') is not None
    critic_loss_config = agent.config["critic_loss"]

    batch_valid = batch['valid'][..., 0]
    for i in range(1, agent.config["horizon_length"]):
        batch_valid = batch_valid * batch['valid'][..., i]
            

    rng, sample_rng = jax.random.split(rng)

    if critic_loss_config['a_next'] == 'BFN':
        next_actions = agent.sample_actions(batch['next_observations'][..., -1, :], rng=sample_rng)
        next_actions = jax.lax.stop_gradient(next_actions)
        next_qs = agent.network.select('target_critic')(batch['next_observations'][..., -1, :], next_actions)
    elif critic_loss_config['a_next'] == 'base':
        from evaluation import sample_base_policy
        base_actions = sample_base_policy(rng, agent, batch['next_observations'][..., -1, :])
        # print(f"base_actions.shape = {base_actions.shape}")  # base_actions.shape = (256, 1, 25)
        base_actions = jnp.squeeze(base_actions, axis=1)
        base_actions = jax.lax.stop_gradient(base_actions)
        next_qs = agent.network.select('target_critic')(batch['next_observations'][..., -1, :], base_actions)
    else:
        raise ValueError(f"Unknown a_next: {critic_loss_config['a_next']}")        

    if critic_loss_config['td_target'] == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_loss_config['td_target'] == 'mean':
        next_q = next_qs.mean(axis=0)
    elif critic_loss_config['td_target'] == 'single':
        idx = jax.random.randint(rng, (), 0, next_qs.shape[0])
        next_q = next_qs[idx]
    elif critic_loss_config['td_target'] == 'double':
        idx = jax.random.choice(rng, next_qs.shape[0], (2,), replace=False)
        next_q = jnp.minimum(next_qs[idx[0]], next_qs[idx[1]])
    else:
        raise ValueError(f"Unknown td_target: {critic_loss_config['td_target']}")

    target_q = batch['rewards'][..., -1] + (agent.config['discount'] ** agent.config["horizon_length"]) * batch['masks'][..., -1] * next_q
    
    q = agent.network.select('critic')(batch['observations'], batch_actions, params=grad_params)
    critic_loss = (jnp.square(q - target_q) * batch_valid).mean()

    # print("q.shape:", q.shape)


    if critic_loss_config.get('cql', None) is not None:
        cql_n_actions = critic_loss_config['cql']['cql_n_actions']

        # Sample actions for CQL, 1: random actions
        action_dim = batch_actions.shape[-1]
        rng, random_action_rng = jax.random.split(rng)
        cql_random_actions = jax.random.uniform(
            random_action_rng, 
            shape=(cql_n_actions, batch['observations'].shape[0], action_dim),
            minval=-1.0 + 1e-5,
            maxval=1.0 - 1e-5,
        )
        cql_random_actions_qs = jnp.stack([
            agent.network.select('critic')(batch['observations'], random_action, params=grad_params) for random_action in cql_random_actions
        ])
        
        # # Sample actions for CQL, 2: policy actions
        rng, policy_action_rng = jax.random.split(rng)
        policy_action_keys = jax.random.split(policy_action_rng, cql_n_actions)
        cql_actions = jnp.stack([
            agent.sample_actions(batch['observations'], rng=key)
            for key in policy_action_keys
        ])
        cql_actions_qs = jnp.stack([
            agent.network.select('critic')(batch['observations'], policy_action, params=grad_params) for policy_action in cql_actions
        ])

        # # Sample actions for CQL, 3: next obs actions
        rng, next_policy_action_rng = jax.random.split(rng)
        next_policy_action_keys = jax.random.split(next_policy_action_rng, cql_n_actions)
        cql_next_actions = jnp.stack([
            agent.sample_actions(batch['next_observations'][..., -1, :], rng=key)
            for key in next_policy_action_keys
        ])
        cql_next_actions_qs = jnp.stack([
            agent.network.select('critic')(batch['observations'], next_action, params=grad_params) for next_action in cql_next_actions
        ])

        print("cql_random_actions_qs.shape:", cql_random_actions_qs.shape)


        cql_cat_q = jnp.concatenate([
            cql_random_actions_qs,
            cql_actions_qs,
            cql_next_actions_qs,
        ], axis=0)

        print("cql_cat_q.shape:", cql_cat_q.shape)

        cql_temperature = critic_loss_config['cql']['cql_temperature']
        cql_min_q_weight = critic_loss_config['cql']['cql_min_q_weight']


        cql_logsumexp = jax.scipy.special.logsumexp(cql_cat_q / cql_temperature, axis=0) * cql_temperature

        cql_q_diff = cql_logsumexp - q

        cql_loss_total = 0.0

        if critic_loss_config['cql'].get('cql_target_action_gap', None) is not None:
            # lagrange def
            # cql_target_action_gap = critic_loss_config['cql']['cql_target_action_gap']
            # cql_alpha_prime = agent.network.select('cql_log_alpha_prime')(params = grad_params)
            # cql_alpha_prime = jnp.clip(jnp.exp(cql_alpha_prime), a_min=0.0, a_max=10.0)

            # print("shape of cql_q_diff:", cql_q_diff.shape)
            # print("shape of cql_alpha_prime:", cql_alpha_prime.shape)
            # print("shape of batch['valid']:", batch['valid'].shape)
            # print("shape of batch_valid:", batch_valid.shape)

            # # alpha loss
            # alpha_loss = -(jax.lax.stop_gradient(cql_q_diff - cql_target_action_gap) * cql_alpha_prime * batch_valid).mean() * cql_min_q_weight
            # cql_loss_total += alpha_loss

            # # cql loss
            # cql_loss = (cql_q_diff * batch_valid).mean() * cql_min_q_weight * jax.lax.stop_gradient(cql_alpha_prime)
            # cql_loss_total += cql_loss
            raise NotImplementedError("CQL target action gap is not implemented yet.")
        else:
            # cql loss
            cql_loss = (cql_q_diff * batch_valid).mean()
            cql_loss_total += cql_loss * cql_min_q_weight
            
                
    info = {}

    if critic_loss_config.get('cql', None) is not None:
        critic_loss = critic_loss + cql_loss_total
        
        info.update({
            'cql_loss': cql_loss_total,
            'cql_logsumexp': cql_logsumexp.mean(),
            # 'cql_alpha_prime': cql_alpha_prime,
        })
        if critic_loss_config['cql'].get('cql_target_action_gap', None) is not None:
            # info['cql_alpha_prime'] = cql_alpha_prime
            raise NotImplementedError("CQL target action gap is not implemented yet.")
        info['critic_loss'] = critic_loss
                
    info.update({
        'critic_loss': critic_loss,
        'q_mean': q.mean(),
        'q_max': q.max(),
        'q_min': q.min(),
    })
        
    return critic_loss, info