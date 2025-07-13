import ml_collections

def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='acrlpd',  # Agent name.
        lr=3e-4,  # Learning rate.
        batch_size=256,  # Batch size.
        actor_hidden_dims=(512, 512, 512, 512),  # Actor network hidden dimensions.
        value_hidden_dims=(512, 512, 512, 512),  # Value network hidden dimensions.
        layer_norm=True,  # Whether to use layer normalization.
        actor_layer_norm=False,  # Whether to use layer normalization for the actor.
        discount=0.99,  # Discount factor.
        tau=0.005,  # Target network update rate.
        num_qs=10,
        target_entropy=ml_collections.config_dict.placeholder(float),  # Target entropy (None for automatic tuning).
        target_entropy_multiplier=0.5,  # Multiplier to dim(A) for target entropy.
        alpha=1.0,
        bc_alpha=0.0,
        q_agg='mean',  # Aggregation function for target Q values.
        horizon_length=ml_collections.config_dict.placeholder(int), # will be set
        action_chunking=True,
        init_temp=1.0,
    ))
    return config