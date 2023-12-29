from typing import *

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

# This is still a abc.ABC, which usually means it is an abstract class | can't be instantiated
class InputEncoding(nn.Module, metaclass=abc.ABCMeta):
    r"""
    Maps input to a flat vector to be fed into neural networks.

    Supports arbitrary batching.
    """
    input_shape: torch.Size
    output_size: int

    def __init__(self, input_shape: torch.Size, output_size: int) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_size = output_size

    @abc.abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass

    # for type hints
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return super().__call__(input)


class Identity(InputEncoding):
    def __init__(self, input_shape: torch.Size) -> None:
        assert len(input_shape) == 1
        super().__init__(input_shape, input_shape.numel())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class OneHot(InputEncoding):
    def __init__(self, input_shape: torch.Size, num_classes: int) -> None:
        assert len(input_shape) == 0, 'we only support single scalar discrete action'
        super().__init__(input_shape, num_classes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dtype == torch.int64
        return F.one_hot(input, self.output_size).to(torch.float32)
    

# Test on Encoding
# 1. InputEncoding
input_tensor = torch.tensor([1,2,3,4,5])
print('--- InputEncoding ---')
print('input_shape: ', input_tensor.shape, ' | type: ', type(input_tensor.shape))
print('numel() -- output_size: ', input_tensor.numel(), ' | type: ', type(input_tensor.numel()))
enc = Identity(input_shape=input_tensor.shape)
print('Identity InputEncoding: ', enc, ' | type: ', type(enc), ' | output_size: ', enc.output_size, '| output_size type: ', type(enc.output_size))
test_tensor = torch.tensor([1,2,3,4,5])
test_out = enc(test_tensor)
print('test_out: ', test_out, ' | type: ', type(test_out))


# 2. OneHot
print('--- OneHot ---')
enc = OneHot(input_shape=torch.Size([]), num_classes=5)
print('OneHot InputEncoding: ', enc, ' | type: ', type(enc), ' | output_size: ', enc.output_size, '| output_size type: ', type(enc.output_size))
test_tensor = torch.tensor([1,2,3,4])
test_out = enc(test_tensor)
print('test_out: ', test_out, ' | type: ', type(test_out))

print('----- Conclusion: Output_size of normally initialized InputEncoding is normal: output_size has type int')


# Test on gym.Space.n
# Basic Carpole-v1
import gym
import numpy as np

# Create an example environment with discrete space
env = gym.make("CartPole-v1")
space = env.action_space

# Check the type of 'n' in the space
space_n_type = type(space.n)
print("Type of 'gym.Space.n':", space_n_type)

# Confirmed to be int, so the issue is not here

import minigrid
import gymnasium as gym

# Using the minigrid environment to test the quasi-metric (needs some asymmetry in the environment)
# -- a small twist makes the task interesting

name = "MiniGrid-FourRooms-v0"
env = gym.make(name, render_mode="rgb_array")
space = env.action_space
space_n_type = type(space.n)
print("Type of 'gym.Space.n':", space_n_type)

