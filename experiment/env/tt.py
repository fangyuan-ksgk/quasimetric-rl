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