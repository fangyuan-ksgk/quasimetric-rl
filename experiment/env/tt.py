import minigrid
import gymnasium as gym
import mediapy

# Using the minigrid environment to test the quasi-metric (needs some asymmetry in the environment)
# -- a small twist makes the task interesting

name = "MiniGrid-FourRooms-v0"
env = gym.make(name, render_mode="rgb_array")

obs, info = env.reset()

while True:
    # sample random action
    action = env.action_space.sample()
    obs, rewards, terms, truncs, info = env.step(action)
    done = terms or truncs
    env.render()
    if done:
        break

print('------------------------------')
print('Test with env.unwrapped.seed()')
# env.seed is deprecated, use env.unwrapped.seed instead
print('._np_random: ', env.unwrapped._np_random)
print('.np_random: ', env.unwrapped.np_random)
# can we reset & change the np_random?
from gymnasium.utils import seeding
seed = 0
set_seed = seeding.np_random(seed)
print('Set seed: ', set_seed[0], ' || from value: ', set_seed[1])
def custom_seed_env(env, seed):
    set_seed = seeding.np_random(seed)
    print('Set seed: ', set_seed[0], ' || from value: ', set_seed[1])
    env.unwrapped._np_random = set_seed[0]
    return env
env = custom_seed_env(env, seed)
print('Custom seeding env -- np_random: ', env.unwrapped.np_random)
