from d4rl.hand_manipulation_suite.door_v0 import DoorEnvV0
from gym.envs.registration import register
import gym, d4rl

class DoorBinaryEnv(DoorEnvV0):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute_reward(self, action, obs):
        # Override to give reward = 1.0 only if door is fully open
        if self.task_success():
            return 0.0
        return -1.0
    
    def get_dataset(self):
        origin_env = gym.make('door-human-v0') 
        return d4rl.qlearning_dataset(origin_env)

class RelocateBinaryEnv(DoorEnvV0):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def compute_reward(self, action, obs):
        # Override to give reward = 1.0 only if object is relocated successfully
        if self.task_success():
            return 0.0
        return -1.0

    def get_dataset(self):
        origin_env = gym.make('relocate-human-v0')
        return d4rl.qlearning_dataset(origin_env)

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