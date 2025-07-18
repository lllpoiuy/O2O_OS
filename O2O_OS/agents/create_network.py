
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from networks.rlpd_networks import Ensemble, StateActionValue, MLP
from networks.rlpd_distributions import TanhNormal
from functools import partial
from networks.flow_networks import ActorVectorField, Value
from networks.encoders import encoder_modules


import jax
import jax.numpy as jnp
import copy
from flax import linen as nn

import optax


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)


def create_normal_network(
    config,
    init_rng,
    ex_observations,
    ex_actions,
):
    """
    Create a normal network for the agent.
    Example config:
    "create_network": {
        "type": "normal"
    }
    """

    action_dim = ex_actions.shape[-1]
    full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
    full_action_dim = full_actions.shape[-1]

    if config["actor_loss"]['target_entropy'] is None:
        config["actor_loss"]['target_entropy'] = -config["actor_loss"]['target_entropy_multiplier'] * full_action_dim

    # Define networks
    critic_base_cls = partial(
        MLP,
        hidden_dims=config['value_hidden_dims'],
        activate_final=True,
        use_layer_norm=config["critic_layer_norm"],
    )
    critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
    critic_def = Ensemble(critic_cls, num=config["num_qs"])


    actor_base_cls = partial(MLP, hidden_dims=config["actor_hidden_dims"], activate_final=True)
    actor_def = TanhNormal(actor_base_cls, full_action_dim)

    # Define the dual alpha variable.
    alpha_def = Temperature(config["actor_loss"]["init_temp"])
    if config["critic_loss"].get("cql", None) is not None:
        if config["critic_loss"]["cql"].get("cql_target_action_gap", None) is not None:
            cql_alpha_def = Temperature(1.0)
        else:
            cql_alpha_def = None
    else:
        cql_alpha_def = None

    network_info = dict(
        critic=(critic_def, (ex_observations, full_actions)),
        target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
        actor=(actor_def, (ex_observations,)),
        alpha=(alpha_def, ()),
    )
    if cql_alpha_def is not None:
        network_info['cql_log_alpha_prime'] = (cql_alpha_def, ())
    networks = {k: v[0] for k, v in network_info.items()}
    network_args = {k: v[1] for k, v in network_info.items()}

    network_def = ModuleDict(networks)
    network_tx = optax.adam(learning_rate=config['lr'])
    network_params = network_def.init(init_rng, **network_args)['params']
    network = TrainState.create(network_def, network_params, tx=network_tx)

    params = network.params
    params['modules_target_critic'] = params['modules_critic']

def create_flow_network(
    config,
    init_rng,
    ex_observations,
    ex_actions,
):
    """
    Create a flow network for the agent.
    Example config:
    "create_network": {
        "type": "flow",
        "flow_steps": 10,
        "use_fourier_features": false,
        "fourier_feature_dim": 64
    }
    """
    
    ex_times = ex_actions[..., :1]
    ob_dims = ex_observations.shape
    action_dim = ex_actions.shape[-1]
    full_actions = jnp.concatenate([ex_actions] * config["horizon_length"], axis=-1)
    full_action_dim = full_actions.shape[-1]

    # Define encoders.
    encoders = dict()
    if config['encoder'] is not None:
        encoder_module = encoder_modules[config['encoder']]
        encoders['critic'] = encoder_module()
        encoders['actor_bc_flow'] = encoder_module()
        encoders['actor_onestep_flow'] = encoder_module()

    # Define networks.
    critic_def = Value(
        hidden_dims=config['value_hidden_dims'],
        layer_norm=config['critic_layer_norm'],
        num_ensembles=config['num_qs'],
        encoder=encoders.get('critic'),
    )

    actor_bc_flow_def = ActorVectorField(
        hidden_dims=config['actor_hidden_dims'],
        action_dim=full_action_dim,
        layer_norm=config['actor_layer_norm'],
        encoder=encoders.get('actor_bc_flow'),
        use_fourier_features=config["create_network"]["use_fourier_features"],
        fourier_feature_dim=config["create_network"]["fourier_feature_dim"],
    )
    actor_onestep_flow_def = ActorVectorField(
        hidden_dims=config['actor_hidden_dims'],
        action_dim=full_action_dim,
        layer_norm=config['actor_layer_norm'],
        encoder=encoders.get('actor_onestep_flow'),
    )

    
    network_info = dict(
        actor_bc_flow=(actor_bc_flow_def, (ex_observations, full_actions, ex_times)),
        actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, full_actions)),
        critic=(critic_def, (ex_observations, full_actions)),
        target_critic=(copy.deepcopy(critic_def), (ex_observations, full_actions)),
    )
    if encoders.get('actor_bc_flow') is not None:
        # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
        network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), (ex_observations,))
    networks = {k: v[0] for k, v in network_info.items()}
    network_args = {k: v[1] for k, v in network_info.items()}

    network_def = ModuleDict(networks)
    network_tx = optax.adam(learning_rate=config['lr'])
    network_params = network_def.init(init_rng, **network_args)['params']
    network = TrainState.create(network_def, network_params, tx=network_tx)

    params = network.params

    params[f'modules_target_critic'] = params[f'modules_critic']

    config['ob_dims'] = ob_dims
    config['action_dim'] = action_dim

    return network