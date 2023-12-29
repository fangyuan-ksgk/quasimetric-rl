import torchqmet
import torch
num_components = 64
iqe = torchqmet.IQE(2048, num_components)
a = torch.randn(2, 2048)
b = torch.randn(2, 2048)
print('shape a: ', a.shape, '| shape b: ', b.shape)
d = iqe(a, b)
print('shape d: ', d.shape)
# Conclusion: IQE computes all components of Interval Quasi-metric Embedding
print(f'Computed {num_components} components of IQE: {d} | Checking gradient pass ...')
loss = d.sum()
loss.backward()
print('Gradient pass enabled')

class L2(torchqmet.QuasimetricBase):
    r"""
    This is a *metric* (not quasimetric) that is used for debugging & comparison.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, num_components=1, guaranteed_quasimetric=True,
                         transforms=[], reduction='sum', discount=None)

    def compute_components(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r'''
        Inputs:
            x (torch.Tensor): Shape [..., input_size]
            y (torch.Tensor): Shape [..., input_size]

        Output:
            d (torch.Tensor): Shape [..., num_components]
        '''
        return (x - y).norm(p=2, dim=-1, keepdim=True)
    
print('Note since torchqmet is C++ extension, it does not support ..(dim=.., components=...) like standard python function interface')
print('-- to change this, create_quasimetric_head_from_spec uses a python function iqe() to wrap the c++ extension')
print('-- to enable string-like config, eval() is adopted to convert string directly to python expression, which gives the IQE model function')
def create_quasimetric_head_from_spec(spec: str) -> torchqmet.QuasimetricBase:
    # Only two are supported
    #   1. iqe(dim=xxx,components=xxx), Interval Quasimetric Embedding
    #   2. l2(dim=xxx), L2 distance

    # IQE divides inputs into k components, each of size l, compute the interval quasimetric embedding on each components, then  
    def iqe(*, dim: int, components: int) -> torchqmet.IQE:
        assert dim % components == 0, "IQE: dim must be divisible by components"
        return torchqmet.IQE(dim, dim // components)

    def l2(*, dim: int) -> L2:
        return L2(dim)

    return eval(spec, dict(iqe=iqe, l2=l2), {})

# test with eval -- replacing key with value and evaluate the python expression (!!)
pickdict = dict(akb='bitches', aespa='whores')
spec = 'akb + akb'
print(eval(spec, pickdict, {}))
print('---------------------------')
# clever way to conver a string into a python expression and initialize a IQE model would be
iqe = create_quasimetric_head_from_spec('iqe(dim=2048, components=64)')
print('Created quasimetric head: ', iqe)

print('IQE is a subclass of torchqmet.QuasimetricBase: ', issubclass(torchqmet.IQE, torchqmet.QuasimetricBase))
print('--- InputSize is a built-in attribute for the torchqmet.QuasimetricBase class: ', iqe.input_size)