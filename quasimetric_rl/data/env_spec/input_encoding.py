from typing import *

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AtariTorso(InputEncoding):

    torso: Callable[[torch.Tensor], torch.Tensor]

    def __init__(self, input_shape: torch.Size, activate_last: bool = True) -> None:

        assert tuple(input_shape) in [(7, 7, 3)]
        super().__init__(input_shape, 512)
        # Now help me calculate the output shape of the torso, input shape is (3, 7, 7)
        # layer 1: (3, 7, 7) -> (32, 3, 3)
        # layer 2: (32, 3, 3) -> (64, 1, 1)
        # layer 3: (64, 1, 1) -> (64, 1, 1)
        self.torso = nn.Sequential(
            nn.Conv2d(self.input_shape[-1], 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

    def permute(self, s: torch.Tensor) -> torch.Tensor:
        assert s.dtype == torch.uint8
        # normalize to [-0.5, 0.5], permute to (N, C, H, W)
        return s.div(255).permute(list(range(s.ndim - 3)) + [-1, -3, -2]) - 0.5

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.torso(self.permute(
            input.flatten(0, -4),
        )).unflatten(0, input.shape[:-3])
