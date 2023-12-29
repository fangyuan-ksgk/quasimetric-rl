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


# -- Configuration is parsed from command-line arguments & Stored
cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='config', node=Conf())

# Learned:
# -- @hydra.main decorator allows for 'train()' calling, which takes 'config' in the StoreConfig object
# -- OmegaConf.to_container()(which is implemented inside .from_DictConfig) converts DictConfig back to the original data type (node type)
# -- Overall, the structure of Conf consists of Actor & Critic
# -- Critic consists of '.model' & '.losses' (Model consists of Encoder, QuasimetricModel, LatentDynamics)
@pdb_if_DEBUG
@hydra.main(version_base=None, config_name="config") 
def train(dict_cfg: DictConfig):

    cfg: Conf = Conf.from_DictConfig(dict_cfg)

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

    # Test with _Expand function on Replay Buffer
    print('---- Testing _Expand function on Replay Buffer ----')
    replay_buffer._expand()

    # Environment's Desired Goal & Target Goal seems to be undefined yet

    # BreakDown of the Conf class all the way to the Encoder
    agent_cfg = cfg.agent # QRLConf
    actor_cfg = agent_cfg.actor # ActorConf
    quasimetric_critic_cfg = agent_cfg.quasimetric_critic # QuasimetricCriticConf
    quasimetric_critic_model_cfg = quasimetric_critic_cfg.model # QuasimetricModelConf
    encoder_cfg = quasimetric_critic_model_cfg.encoder # EncoderConf
    latent_dynamics_cfg = quasimetric_critic_model_cfg.latent_dynamics # LatentDynamicsConf
    quasimetric_cfg = quasimetric_critic_model_cfg.quasimetric_model # QuasimetricModelConf

    # Make an encoder with the encoder_cfg -- with env_spec
    print('Test on Encoder')
    encoder = encoder_cfg.make(env_spec=replay_buffer.env_spec)
    print('Encoder make success: ')
    print(encoder)




if __name__ == '__main__':
    if 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    # set up some hydra flags before parsing
    os.environ['HYDRA_FULL_ERROR'] = str(int(FLAGS.DEBUG))

    # check the mysterious dict_config
    # print('Checking the teleported dict_config')
    # print(dict_config)
    train()