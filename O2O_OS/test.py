import gym, d4rl
import envs.d4rl_binary

env = gym.make('door-binary-v0')
dataset = d4rl.qlearning_dataset(env)

env = gym.make('relocate-binary-v0')
dataset = d4rl.qlearning_dataset(env)

import gymnasium
import shimmy
gym_env = gym.make("relocate-human-v1")
env = gymnasium.make('GymV21Environment-v0', env_id="relocate-human-v1")
obs = env.reset()
dataset = d4rl.qlearning_dataset(gym_env)