from typing import *

import attrs

import torch
import torch.nn as nn

from ...utils import MLP, LatentTensor

from ....data import EnvSpec
from ....data.env_spec.input_encoding import InputEncoding


# LatentDynamics is a MLP that takes in a latent vector and an action vector
class LatentDynamics(MLP):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior

        arch: Tuple[int, ...] = (512, 512)
        residual: bool = True

        def make(self, *, latent_size: int, env_spec: EnvSpec) -> 'LatentDynamics':
            return LatentDynamics(
                latent_size=latent_size,
                env_spec=env_spec,
                hidden_sizes=self.arch,
                residual=self.residual,
            )

    action_input: InputEncoding
    residual: bool

    def __init__(self, *, latent_size:int, env_spec: EnvSpec, hidden_sizes: Tuple[int, ...], residual: bool):
        action_input = env_spec.make_action_input()
        print('----- Begining of LatentDynamics initialization -----')
        inputs1 = latent_size + action_input.output_size
        inputs2 = latent_size
        inputs3 = hidden_sizes
        inputs4 = residual
        print('---- Check on initialization values for LatentDynamics ----')
        print('action_input.output_size: ', action_input.output_size, ' | Type: '  , type(action_input.output_size))
        

        print('---- Issue Spotted: ----', 'action_input.output_size is a numpy.int64')
        print('---- Converting type to int ----')
        action_input.output_size = int(action_input.output_size)
        print('latent_size + action_input.output_size: ', inputs1, ' | Type: '  , type(inputs1))
        print('latent_size: ', inputs2, ' | Type: '  , type(inputs2))
        print('hidden_sizes: ', inputs3, ' | Type: '  , type(inputs3))
        print('residual: ', inputs4, ' | Type: '  , type(inputs4))

        # print('-- Did not see any numpy.int64 here --')
        print('-- Begin super().__init__ going into MLP initialization steps')
        super().__init__(
            latent_size + action_input.output_size,
            latent_size,
            hidden_sizes=hidden_sizes,
            zero_init_last_fc=residual,
        )
        self.action_input = action_input
        self.residual = residual
        print('---- line44. Sucess in initialization of LatentDynamics')

    def forward(self, zx: LatentTensor, action: torch.Tensor) -> LatentTensor:
        # broadcast batch shapes before cat
        action = self.action_input(action)
        broadcast_bshape: torch.Size = torch.broadcast_shapes(zx.shape[:-1], action.shape[:-1])
        zx = zx.expand(broadcast_bshape + zx.shape[-1:])
        action = action.expand(broadcast_bshape + action.shape[-1:])

        zy = super().forward(
            torch.cat([zx, action], dim=-1)
        )
        if self.residual:
            zy = zx + zy
        return zy

    # for type hints
    def __call__(self, zx: LatentTensor, action: torch.Tensor) -> LatentTensor:
        return nn.Module.__call__(self, zx, action)

    def extra_repr(self) -> str:
        return super().extra_repr() + f"\nresidual={self.residual}"
