from d4rl.hand_manipulation_suite.door_v0 import DoorEnvV0
from gym.envs.registration import register
import gym, d4rl

class DoorBinaryEnv(DoorEnvV0):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def step(self, action):
        observation, reward, terminated, truncated = super().step(action)
        if truncated['goal_achieved']:
            reward = 0.0
        else:
            reward = -1.0
        return observation, reward, terminated, truncated
    
    def get_dataset(self):
        origin_env = gym.make('door-human-v0') 
        return d4rl.qlearning_dataset(origin_env)

    def get_normalized_score(self, score):
        """Return normalized score between 0 and 1"""
        return 1.0 + score/100.0  # Normalized to [0, 1]


class RelocateBinaryEnv(DoorEnvV0):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def step(self, action):
        observation, reward, terminated, truncated = super().step(action)
        if truncated['goal_achieved']:
            reward = 0.0
        else:
            reward = -1.0
        return observation, reward, terminated, truncated

    def get_dataset(self):
        origin_env = gym.make('relocate-human-v0')
        return d4rl.qlearning_dataset(origin_env)

    def get_normalized_score(self, score):
        """Return normalized score between 0 and 1"""
        return 1.0 + score/100.0  # Normalized to [0, 1]


register(
    id='relocate-binary-v0',
    entry_point='envs.d4rl_binary:RelocateBinaryEnv',
    max_episode_steps=100,
)

register(
    id='door-binary-v0',
    entry_point='envs.d4rl_binary:DoorBinaryEnv',
    max_episode_steps=100,
)