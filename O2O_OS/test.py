import gym
import d4rl
import gymnasium
import shimmy
gym_env = gym.make("relocate-human-v1")
env = gymnasium.make('GymV21Environment-v0', env_id="relocate-human-v1")
obs = env.reset()
dataset = d4rl.qlearning_dataset(gym_env)