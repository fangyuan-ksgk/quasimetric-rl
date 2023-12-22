import gymnasium as gym
env = gym.make('MountainCar-v0', render_mode='human')
obs, info = env.reset()

while True:
    # sample random action
    action = env.action_space.sample()
    obs, rewards, terms, truncs, info = env.step(action)
    done = terms or truncs
    env.render()
    if done:
        break