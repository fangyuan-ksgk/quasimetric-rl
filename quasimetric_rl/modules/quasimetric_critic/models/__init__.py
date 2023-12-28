from typing import *

import attrs

import torch
import torch.nn as nn

from .encoder import Encoder
from .quasimetric_model import QuasimetricModel
from .latent_dynamics import LatentDynamics

from ...utils import Module

from ....data import EnvSpec

# This is a Great Breaking point into the codebase
class QuasimetricCritic(Module):
    @attrs.define(kw_only=True)
    class Conf:
        # config / argparse uses this to specify behavior
        # Inside the Conf class, everything is actually a Conf object
        # -- e.g. encoder is actually the conf object of the Encoder class
        # Encoder.Conf tells us that the encoder's Type || Encoder.Conf() initialize a instance of the Type, giving a default value for it.
        encoder: Encoder.Conf = Encoder.Conf()
        quasimetric_model: QuasimetricModel.Conf = QuasimetricModel.Conf()
        latent_dynamics: LatentDynamics.Conf = LatentDynamics.Conf()

        def make(self, *, env_spec: EnvSpec) -> 'QuasimetricCritic':
            # High-level Intuition
            # Information of Rd latent space is parametrized in latent_size
            # Information of observation, action in env is included in env_spec
            # -- Encoder needs information on the env observation
            # -- QuasimetricModel operates only on latent space
            # -- LatentDynamics predict next latent state given current latent state and action
            encoder = self.encoder.make(
                env_spec=env_spec,
            )
            quasimetric_model = self.quasimetric_model.make(
                input_size=encoder.latent_size,
            )
            latent_dynamics = self.latent_dynamics.make(
                latent_size=encoder.latent_size,
                env_spec=env_spec,
            )
            return QuasimetricCritic(encoder, quasimetric_model, latent_dynamics)

    encoder: Encoder
    quasimetric_model: QuasimetricModel
    latent_dynamics: LatentDynamics

    raw_lagrange_multiplier: nn.Parameter  # for the QRL constrained optimization


    def __init__(self, encoder: Encoder, quasimetric_model: QuasimetricModel, latent_dynamics: LatentDynamics):
        super().__init__()
        self.encoder = encoder
        self.quasimetric_model = quasimetric_model
        self.latent_dynamics = latent_dynamics

    def forward(self, x: torch.Tensor, y: torch.Tensor, *, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        # The basic interface is a V- or Q-function.
        zx = self.encoder(x)
        zy = self.encoder(y)
        if action is not None:
            zx = self.latent_dynamics(zx, action)
        return self.quasimetric_model(zx, zy)

    # for type hints
    def __call__(self, x: torch.Tensor, y: torch.Tensor, *, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        return super().__call__(x, y, action=action)
