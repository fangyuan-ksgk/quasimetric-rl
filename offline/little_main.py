from typing import *

import os

import glob
import attrs
import logging
import time

from omegaconf import DictConfig
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.backends.cudnn
import torch.multiprocessing

import quasimetric_rl
from quasimetric_rl import utils, pdb_if_DEBUG, FLAGS

from quasimetric_rl.utils.steps_counter import StepsCounter
from quasimetric_rl.modules import InfoT
from quasimetric_rl.base_conf import BaseConf

from .trainer import Trainer

import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path='../experiment/', config_name='exp_cfg1')
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()