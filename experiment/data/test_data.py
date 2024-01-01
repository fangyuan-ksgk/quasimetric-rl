# test with DataBatch and base class
import abc
import attrs
import attr
import torch
import torch.utils.data
from typing import *
from typing_extensions import Self, get_args, get_origin

# no Attribute is defined with attrs.s decorator
# @attrs.define(kw_only=True)
class TensorCollectionAttrsMixin(abc.ABC):
    a : int = 1
    b : float = attr.ib(0.444)
    c : float = attrs.field(default=0.999)
    @classmethod
    def types_dict(cls):
        fields = attrs.fields_dict(cls) # able to access all attrs decorated attributes
        print('fields_dict: ', fields)
        print('type hints: ', get_type_hints(cls)) # able to access all attributes in current class
        return {k: t for k, t in get_type_hints(cls).items() if k in fields}
    
@attrs.define(kw_only=True)
class DB(TensorCollectionAttrsMixin):
    pass

print(DB.types_dict())
# print(data.types_dict())