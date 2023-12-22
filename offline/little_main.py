from typing import *

import os

import glob
import attrs
import logging
import time

from omegaconf import DictConfig
import hydra
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

@utils.singleton
@attrs.define(kw_only=True)
class Conf(BaseConf):
    output_base_dir: str = attrs.field(default=os.path.join(os.path.dirname(__file__), 'results'))

    resume_if_possible: bool = False

    env: quasimetric_rl.data.Dataset.Conf = quasimetric_rl.data.Dataset.Conf()

    batch_size: int = attrs.field(default=4096, validator=attrs.validators.gt(0))
    num_workers: int = attrs.field(default=8, validator=attrs.validators.ge(0))
    total_optim_steps: int = attrs.field(default=int(2e5), validator=attrs.validators.gt(0))

    log_steps: int = attrs.field(default=250, validator=attrs.validators.gt(0))
    save_steps: int = attrs.field(default=50000, validator=attrs.validators.gt(0))



cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='config', node=Conf())

# There is no reason why we would want to use other environments than the ones we have

@pdb_if_DEBUG
@hydra.main(version_base=None, config_name="config")
def train(dict_cfg: DictConfig):
    cfg: Conf = Conf.from_DictConfig(dict_cfg)
    writer = cfg.setup_for_experiment()  # checking & setup logging
    print(f'{("").join(["-"]*31)} Making Environment {("").join(["-"]*31)}')
    dataset = cfg.env.make() # Make the Environment -- seems to be problematic here


if __name__ == '__main__':
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    # set up some hydra flags before parsing
    os.environ['HYDRA_FULL_ERROR'] = str(int(FLAGS.DEBUG))

    train()