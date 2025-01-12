from typing import *

import attrs

import torch
import torch.nn as nn

from ...utils import MLP, LatentTensor

from ....data import EnvSpec
from ....data.env_spec.input_encoding import InputEncoding

# More than 3 questions needs to be asked here:
# 1. What is the purpose of the attrs.define decorator?
# 2. What is the purpose of the Conf class?
# 3. What is the purpose of the __call__ method?
class Encoder(nn.Module):
    r"""
    (*, *input_shape)                      Input
           |
     [input_encoding]                      e.g., AtariTorso network to map image input to a flat vector
           |
        (*, d)                             Encoded 1-D input
           |
    [MLP specified by arch]                ENCODER
           |
        (*, latent_size)                   1-D Latent
    """

    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        arch: Tuple[int, ...] = (512, 512)
        latent_size: int = 128

        def make(self, *, env_spec: EnvSpec) -> 'Encoder':
            return Encoder(
                env_spec=env_spec,
                arch=self.arch,
                latent_size=self.latent_size,
            )

    input_shape: torch.Size
    input_encoding: InputEncoding
    encoder: MLP
    latent_size: int

    def __init__(self, *, env_spec: EnvSpec,
                 arch: Tuple[int, ...], latent_size: int, **kwargs):
        super().__init__(**kwargs)
        # Encoder relies on MLP and InputEncoding
        self.input_shape = env_spec.observation_shape
        self.input_encoding = env_spec.make_observation_input()
        encoder_input_size = self.input_encoding.output_size
        self.encoder = MLP(encoder_input_size, latent_size, hidden_sizes=arch)
        self.latent_size = latent_size

    def forward(self, x: torch.Tensor) -> LatentTensor:
        return self.encoder(self.input_encoding(x))

    # for type hint (we test on this!)
    def __call__(self, x: torch.Tensor) -> LatentTensor:
        return super().__call__(x)

    def extra_repr(self) -> str:
        return f"input_shape={self.input_shape}, latent_size={self.latent_size}"
