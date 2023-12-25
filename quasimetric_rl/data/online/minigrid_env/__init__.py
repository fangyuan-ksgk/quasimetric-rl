from typing import *

import functools

import minigrid
import gymnasium as gym
import gymnasium.spaces
import numpy as np

from ..memory import register_online_env

# Remains to be verified
class GoalCondEnvWrapper(gym.ObservationWrapper):
    r"""
    Convert the concatenated observation space in GCRL into a better format with
    dict observations.
    """

    episode_length: int
    is_image_based: bool
    create_kwargs: Mapping[str, Any]

    def __init__(self, env: gym.Env, episode_length: int, is_image_based: bool = False):
        super().__init__(gym.wrappers.TimeLimit(env.unwrapped, episode_length))
        if is_image_based:
            # Observation space assumption : shape is (7, 7, 3)
            single_ospace = gym.spaces.Box(
                low=np.full((7, 7, 3), 0),
                high=np.full((7, 7, 3), 255),
                dtype=np.uint8,
            )
        else:
            assert isinstance(env.observation_space, gym.spaces.Box)
            ospace: gym.spaces.Box = env.observation_space
            assert len(ospace.shape) == 1
            single_ospace = gym.spaces.Box(
                low=np.split(ospace.low, 2)[0],
                high=np.split(ospace.high, 2)[0],
                dtype=ospace.dtype,
            )
        self.observation_space = gym.spaces.Dict(dict(
            observation=single_ospace,
            achieved_goal=single_ospace,
            desired_goal=single_ospace,
        ))
        self.episode_length = episode_length
        self.is_image_based = is_image_based

    def observation(self, observation):
        o, g = np.split(observation, 2)
        if self.is_image_based:
            o = o.reshape(7, 7, 3)
            g = g.reshape(7, 7, 3) # Wait, g is the goal map? what does that mean?
        odict = dict(
            observation=o,
            achieved_goal=o,
            desired_goal=g,
        )
        return odict
    
# Test version: Wrap Minigrid environment into ImageObservation first, then to GoalCondEnvWrapper
def create_env_from_spec(name: str):
    env: gym.Env = minigrid.wrappers.ImgObsWrapper(gym.make(name, render_mode='rgb_array'))
    is_image_based = True
    return GoalCondEnvWrapper(env, episode_length=50, is_image_based=is_image_based)

valid_names = (
    'MiniGrid-FourRooms-v0',
    'MiniGrid-OneWayDoor-v0',
)

for name in valid_names:
    register_online_env('minigrid_env', name, 
                         create_env_fn=functools.partial(create_env_from_spec, name=name),
                         episode_length=50)