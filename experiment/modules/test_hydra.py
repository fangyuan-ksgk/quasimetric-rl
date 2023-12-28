# Test on Hydra & OmegaConf
import abc
import tempfile
from typing import *
import attrs
import hydra
import hydra.types
import hydra.core.config_store
from omegaconf import OmegaConf, DictConfig, SCMode

# Overall the tricks with the Hydra is in the Decorator -- it allows implicit dictConfig Input without having to pass any argument explicitly.
# --- either through reading .yaml file, or using some ConfigStore instance, just change the @hydra.main decorator and things will be fine

@attrs.define(kw_only=True)
class BaseConf(abc.ABC):
    a: int = 1
    b: int = 2
    
    @classmethod
    def from_DictConfig(cls, cfg: DictConfig) -> 'BaseConf':
        return OmegaConf.to_container(cfg, structured_config_mode=SCMode.INSTANTIATE)

@attrs.define(kw_only=False)
class Conf(BaseConf):
    pass

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='pig', node=Conf())

@hydra.main(version_base=None, config_name="pig")
def test(dict_config: DictConfig):
    cfg: Conf = Conf.from_DictConfig(dict_config)
    print('Obtained Config: ', cfg)

test()