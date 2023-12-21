import hydra
from omegaconf import OmegaConf, DictConfig
base_path = '/Users/fangyuanyu/Implementation/Agent/quasimetric-rl'
@hydra.main(version_base=None, config_path=f'{base_path}/experiment/', config_name='config')
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()