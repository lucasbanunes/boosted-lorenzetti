import torch.nn as nn
from typing import List
import torch


from ..models.torch import torch_module_from_string


def build_mlp(
    dims: List[int],
    activations: List[str | None]
):
    model = nn.Sequential()
    iterator = zip(dims[:-1],
                   dims[1:],
                   activations)
    for input_dim, output_dim, activation in iterator:
        model.append(nn.Linear(input_dim, output_dim))
        if activation is not None:
            model.append(torch_module_from_string(activation))
    model.example_input_array = torch.randn(dims[0])
    return model
