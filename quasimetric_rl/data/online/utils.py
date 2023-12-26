from __future__ import annotations
from typing import *

import numpy as np
import torch
import torch.utils.data

from .. import EnvSpec
from ..base import (
    EpisodeData, MultiEpisodeData,
)


def get_empty_episode(env_spec: EnvSpec, episode_length: int) -> EpisodeData:
    r'''
    episode_lengths: torch.Tensor                                          # [1]
    all_observations: torch.Tensor                                        # [L + 1, *observation_shape]
    actions: torch.Tensor                                                 # [L, *action_shape]
    rewards: torch.Tensor                                                 # [L]
    terminals: torch.Tensor                                               # [L]
    timeouts: torch.Tensor                                                # [L]
    observation_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)   # [L + 1]
    transition_infos: Mapping[str, torch.Tensor] = attrs.Factory(dict)    # [L]

    Only needs to fill in
        all_observations
        actions
        rewards
        is_success
        desired_goals
    afterwards.
    '''
    all_observations = torch.empty(episode_length + 1, *env_spec.observation_shape, dtype=env_spec.observation_dtype)
    timeouts = torch.zeros(episode_length, dtype=torch.bool)
    timeouts[-1] = True
    if env_spec.observation_space_is_dict:
        observation_infos = dict(
            achieved_goals=torch.empty_like(all_observations),
            desired_goals=torch.empty_like(all_observations),
        )
    else:
        observation_infos = {}
    return EpisodeData(
        episode_lengths=torch.tensor(episode_length).view(-1),
        all_observations=all_observations,
        actions=torch.empty(episode_length, *env_spec.action_shape, dtype=env_spec.action_dtype),
        rewards=torch.empty(episode_length),
        terminals=torch.zeros(episode_length, dtype=torch.bool),
        timeouts=timeouts,
        observation_infos=observation_infos,
        transition_infos=dict(
            is_success=torch.empty(episode_length, dtype=torch.bool),  # not used by QRL, but fill in if you want to know in loaded batch
        ),
    )


def get_empty_episodes(env_spec: EnvSpec, episode_length: int, num_episodes: int) -> MultiEpisodeData:
    # Question: Why do we need to add num_episodes to the first dimension of all_observations?
    # Answer: Because we are going to flatten the first dimension of all_observations later on.
    # Question: Why is the flatten step requires an extra episode?
    # Answer: Because we are going to use the last episode as the first episode of the next batch.
    # Question: What? Next Batch? Why is it needed? And why is the current batch then not subtracted by 1 episode, since it is already represented in the previous batch?
    # Answer: Because we are going to use the last episode as the first episode of the next batch.
    all_observations = torch.empty(episode_length * num_episodes + num_episodes, *env_spec.observation_shape, dtype=env_spec.observation_dtype)
    timeouts = torch.zeros(num_episodes, episode_length, dtype=torch.bool)
    timeouts[:, -1] = True
    if env_spec.observation_space_is_dict:
        observation_infos = dict(
            achieved_goals=torch.empty_like(all_observations),
            desired_goals=torch.empty_like(all_observations),
        )
    else:
        observation_infos = {}
    # Question: Is it true that by definition, MultiEpisodeData requires an extra episode in the 'all_observation' tensor?
    # Answer: Yes, it is true. See the definition of MultiEpisodeData.
    return MultiEpisodeData(
        episode_lengths=torch.full([num_episodes], episode_length, dtype=torch.int64),
        all_observations=all_observations,
        actions=torch.empty(episode_length * num_episodes, *env_spec.action_shape, dtype=env_spec.action_dtype),
        rewards=torch.empty(episode_length * num_episodes),
        terminals=torch.zeros(episode_length * num_episodes, dtype=torch.bool),
        timeouts=timeouts.flatten(),
        observation_infos=observation_infos,
        transition_infos=dict(
            is_success=torch.empty(episode_length * num_episodes, dtype=torch.bool),  # not used by QRL, but fill in if you want to know in loaded batch
        ),
    )
