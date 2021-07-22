"""
Backbone modules.
"""
from torch import nn
from util.misc import NestedTensor
from .position_encoding import build_position_encoding


class Joiner(nn.Sequential):
    def __init__(self, position_embedding):
        super().__init__(position_embedding)

    def forward(self, tensor_list: NestedTensor):
        pos = []
        # position encoding
        pos.append(self[0](tensor_list).to(tensor_list.tensors.dtype))
        return [tensor_list], pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    model = Joiner(position_embedding)
    model.num_channels = 18
    return model
