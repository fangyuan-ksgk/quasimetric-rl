from typing import *

import os
import collections
import logging
import attrs
import abc
from tqdm.auto import tqdm

import numpy as np
import torch
import gymnasium as gym

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)

# Global Variable Type
OfflineEnv = None

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

# Great efforts from author is put into working around the un-compatible d4rl package
# I do not need that

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name: Union[str, gym.Env]) -> gym.Env:
    if type(name) != str:
        ## name is already an environment
        return name
    # when name is a simple string, use gym.make(name) to create the environment here
    with suppress_output():
        wrapped_env: gym.Wrapper = gym.make(name)
    # env = wrapped_env.unwrapped
    env: 'OfflineEnv' = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    env.reset()
    env.step(env.action_space.sample())  # sometimes stepping is needed to initialize internal
    env.reset()
    return env






