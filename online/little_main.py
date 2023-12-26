from typing import *

import os

import attrs
import logging
import json
import time

import hydra
import hydra.types
import hydra.core.config_store
from omegaconf import DictConfig

import torch
import torch.backends.cudnn
import torch.multiprocessing

import quasimetric_rl
from quasimetric_rl import utils, pdb_if_DEBUG, FLAGS

from quasimetric_rl.utils.steps_counter import StepsCounter
from quasimetric_rl.modules import InfoT
from quasimetric_rl.base_conf import BaseConf

from .trainer import Trainer, InteractionConf


@utils.singleton
@attrs.define(kw_only=True)
class Conf(BaseConf):
    output_base_dir: str = attrs.field(default=os.path.join(os.path.dirname(__file__), 'results'))

    env: quasimetric_rl.data.online.ReplayBuffer.Conf = quasimetric_rl.data.online.ReplayBuffer.Conf()

    batch_size: int = attrs.field(default=256, validator=attrs.validators.gt(0))
    interaction: InteractionConf = InteractionConf()

    log_steps: int = attrs.field(default=250, validator=attrs.validators.gt(0))
    eval_steps: int = attrs.field(default=2000, validator=attrs.validators.gt(0))
    save_steps: int = attrs.field(default=50000, validator=attrs.validators.gt(0))



cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='config', node=Conf())

@pdb_if_DEBUG
@hydra.main(version_base=None, config_name="config")
def train(dict_cfg: DictConfig):
    # Even though from_dictconfig pass all the configs input into the Conf object, it is still strange
    # -- how the config pass into the env.Conf object?
    cfg: Conf = Conf.from_DictConfig(dict_cfg)
    # All the configurations are set here -- fromDictConfig clearly passes env.kind/name onto cfg such that
    # -- values of cfg.env.kind/name are set ... obviously ...
    # print('Checking on the Conf Object')
    # print('--- Type: ', type(cfg))
    # print('--- Attributes env.kind', cfg.env.kind)
    # print('--- Attributes env.name', cfg.env.name)
    # print('--- Attributes type(env)', type(cfg.env))


    writer = cfg.setup_for_experiment()  # checking & setup logging
    replay_buffer = cfg.env.make()

    print('--- Attribute Shape Checker ---')
    print('--- replay_buffer.episode_length: ', replay_buffer.episode_length)
    print('--- replay_buffer.num_episodes_realized: ', replay_buffer.num_episodes_realized)
    print('--- replay_buffer.episode_capacity(): ', replay_buffer.episodes_capacity)
    print('--- replay_buffer.raw_data type: ', type(replay_buffer.raw_data))
    print('--- replay_buffer.raw_data.episode_lengths shape: ', replay_buffer.raw_data.episode_lengths.shape)
    print('--- replay_buffer.raw_data.episode_lengths[:5]: ', replay_buffer.raw_data.episode_lengths[:5])

    print('--- replay_buffer.raw_data.all_observation shape: ', replay_buffer.raw_data.all_observations.shape)
    print('--- replay_buffer.raw_data.actions shape: ', replay_buffer.raw_data.actions.shape)
    print('--- replay_buffer.raw_data.rewards shape: ', replay_buffer.raw_data.rewards.shape)
    print('--- replay_buffer.raw_data.terminals shape: ', replay_buffer.raw_data.terminals.shape)
    print('--- replay_buffer.raw_data.timeouts shape: ', replay_buffer.raw_data.timeouts.shape)
    print('--- replay_buffer.raw_data.observation_infos keys: ', replay_buffer.raw_data.observation_infos.keys())

    # print('Agent Configuration: ', cfg.agent)

    # trainer = Trainer(
    #     agent_conf=cfg.agent,
    #     device=cfg.device.make(),
    #     replay=replay_buffer,
    #     batch_size=cfg.batch_size,
    #     interaction_conf=cfg.interaction,
    # )

if __name__ == '__main__':
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    # set up some hydra flags before parsing
    os.environ['HYDRA_FULL_ERROR'] = str(int(FLAGS.DEBUG))

    train()