from __future__ import annotations
from typing import *

import logging
import functools

import numpy as np
import torch.utils.data
import gym
import gym.spaces

from ..base import register_offline_env
from . import load_environment, convert_dict_to_EpisodeData_iter, sequence_dataset


if TYPE_CHECKING:
    import d4rl.pointmaze


# Environment does NOT have a Policy right? So how can we obtain dataset without knowing how to act?
# I think we need to generate episodes from the gym environment

def preprocess_maze2d_fix(env: 'd4rl.pointmaze.MazeEnv', dataset: Mapping[str, np.ndarray]):
    # If I can reach the goal, I will not need to run algorithm .....
    ## In generation, controller is run until reached goal, which is
    ## continuously set.
    ##
    ## There, terminal is always False, and how timeout is set is unknown (not
    ## in the public script)
    ##
    ## Set timeout at time t                      (*)
    ##   iff reached goal at time t
    ##   iff goal at time t != goal at time t+1
    ##
    ## Remove terminals
    ##
    ## Add next_observations
    ##
    ## Also Maze2d *rewards* is field is off-by-one:
    ##    rewards[t] is not the reward received for performing actions[t] at observation[t].
    ## Rather, it is the reward to be received for transitioning *into* observation[t].
    ##
    ## What a mess... This fixes that too.
    ##
    ## NB that this is different from diffuser code!

    # dataset keys requirements: -- we need only construct such a dataset and that is it
    #  observations: (N, 2) float32 array
    #  actions: (N, 2) float32 array
    #  rewards: (N,) float32 array
    #  terminals: (N,) bool array
    #  infos/goal: (N, 2) float32 array
    #  infos/start: (N, 2) float32 array
    #  infos/success: (N,) bool array
 
    assert not np.any(dataset['terminals'])
    dataset['next_observations'] = dataset['observations'][1:]

    goal_diff = np.abs(dataset['infos/goal'][:-1] - dataset['infos/goal'][1:]).sum(-1)  # diff with next
    timeouts = goal_diff > 1e-5

    timeout_steps = np.where(timeouts)[0]
    path_lengths = timeout_steps[1:] - timeout_steps[:-1]

    logging.info(
        f'[ preprocess_maze2d_fix ] Segmented {env.name} | {len(path_lengths)} paths | '
        f'min length: {path_lengths.min()} | max length: {path_lengths.max()}'
    )

    dataset['timeouts'] = timeouts

    logging.info('[ preprocess_maze2d_fix ] Fixed terminals and timeouts')

    # Fix rewards
    assert len(env.goal_locations) == 1
    rewards = cast(
        np.ndarray,
        np.linalg.norm(
            dataset['next_observations'][:, :2] - env.get_target(),
            axis=-1,
        ) <= 0.5
    ).astype(dataset['rewards'].dtype)
    # check that it was wrong :/
    assert (rewards == dataset['rewards'][1:]).all()
    dataset['rewards'] = rewards
    logging.info('[ preprocess_maze2d_fix ] Fixed rewards')

    # put things back into a new dict
    dataset = dict(dataset)
    for k in dataset:
        if dataset[k].shape[0] != dataset['next_observations'].shape[0]:
            dataset[k] = dataset[k][:-1]
    return dataset


# My guess is that the so-called dataset is precisely a replay-buffer, which collects historical datas
# -- proof is there is not even a policy in the environment
def load_episodes_maze2d(name):
    env = load_environment(name)
    # yield returns a generator
    # yield from operates on a generator (defined through yield)
    yield from convert_dict_to_EpisodeData_iter(
        sequence_dataset(
            env,
            preprocess_maze2d_fix(
                env,
                env.get_dataset(), # This is weird by definition - what is the policy?? wait, unless it's excatly
            ),
        ),
    )


for name in ['maze2d-umaze-v1', 'maze2d-medium-v1', 'maze2d-large-v1']:
    register_offline_env(
        'd4rl', name,
        create_env_fn=functools.partial(load_environment, name),
        load_episodes_fn=functools.partial(load_episodes_maze2d, name),
    )


