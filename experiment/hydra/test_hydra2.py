from omegaconf import DictConfig, OmegaConf
import hydra
print('Absolutely no structure, Empty Configurations!')
print('Hydra can use decorator to allow real-time addition of configuration variables & values: use +name=value')
@hydra.main(version_base=None)
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()