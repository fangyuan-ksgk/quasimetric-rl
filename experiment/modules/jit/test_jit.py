from typing import *
import torch
import torch.nn as nn
import numpy as np

# By using Type[nn.Module], the hint suggests that activation_fn should be a class 
# that is either nn.Module itself or a subclass of nn.Module.

# -- Type[class] helps clarify the argument should be a class, and not an instance of the class
# -- here, nn.ReLU is the default class, note that nn.ReLU() would be an instance and will not be acceptable

# R: initialize a nn.moduels and use torch.jit to convert it
# -- reproduce the issue
class MLP(nn.Module):
    input_size: int
    output_size: int
    zero_init_last_fc: bool
    module: nn.Sequential

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 *,
                 hidden_sizes: Collection[int],
                 activation_fn: Type[nn.Module] = nn.ReLU, # basically activation_fn should be a Class, not an instance
                 zero_init_last_fc: bool = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.zero_init_last_fc = zero_init_last_fc

        layer_in_size = input_size
        modules: List[nn.Module] = []
        for sz in hidden_sizes:
            modules.extend([
                nn.Linear(layer_in_size, sz),
                activation_fn(),
            ])
            layer_in_size = sz
        modules.append(
            nn.Linear(layer_in_size, output_size),
        )

        # initialize with glorot_uniform
        with torch.no_grad():
            def init_(m: nn.Module):
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            for m in modules:
                m.apply(init_)
            if zero_init_last_fc:
                last_fc = cast(nn.Linear, modules[-1])
                last_fc.weight.zero_()
                last_fc.bias.zero_()

        print('--- modules/utils.py --- Line 111 --- MLP ---')
        print('We have a type issue here with the modules |' ,)
        self.module = torch.jit.script(nn.Sequential(*modules))
        print('Jit script success')

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.module(input)

    # for type hints
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return super().__call__(input)

    def extra_repr(self) -> str:
        return "zero_init_last_fc={}".format(
            self.zero_init_last_fc,
        )
    
print('Check on type of nn.Relu: ', type(nn.ReLU), '-- is nn.ReLU a subclass of nn.Module?: --', issubclass(nn.ReLU, nn.Module), '--')

# initialize MLP
print('Initialize MLP')
mlp = MLP(input_size=2, output_size=2, hidden_sizes=[2,2])
print(mlp.module)

print('MLP can be initialized without any problem')
print('Issue is probably with the argument for the initialization of MLP | type error there')
