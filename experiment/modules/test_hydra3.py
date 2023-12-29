import hydra
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig, SCMode
from hydra.core.config_store import ConfigStore


@dataclass
class MySQLConfig:
    host: str = "localhost"
    port: int = 3306

cs = ConfigStore.instance()

# Using the type
cs.store(name="config1", node=MySQLConfig)
# Using an instance, overriding some default values
cs.store(name="config2", node=MySQLConfig(host="test.db", port=3307))
# Using a dictionary, forfeiting runtime type safety
cs.store(name="config3", node={"host": "localhost", "port": 3308})

@hydra.main(config_name="config2") # I am using the stored config named 'config3'
def whatever(dict_config: DictConfig):
    print('Decorator automatically reads the stored ConfigStore and passes the DictConfig to the function')
    print(type(dict_config), dict_config)
    print('---------')
    print('OmegaConf.to_container to process the DictConfig variable:')
    print('--- It converts DictConfig back to the original data type (node type) ---')
    cfg = OmegaConf.to_container(dict_config, structured_config_mode=SCMode.INSTANTIATE)
    print(type(cfg), cfg)

whatever()