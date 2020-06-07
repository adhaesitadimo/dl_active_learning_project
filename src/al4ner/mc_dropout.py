from typing import Tuple

import torch
from torch import nn


class DropoutMC(nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x):
        return nn.functional.dropout(x, self.p, training=self.training or self.activate)


def convert_to_mc_dropout(model: nn.Module, layers_to_substitute: Tuple[nn.Module] = (nn.Dropout,)):
    for i, layer in enumerate(list(model.children())):
        if isinstance(layer, layers_to_substitute):
            name = list(model._modules.items())[i][0]
            model._modules[name] = DropoutMC(p=layer.p, activate=True)
        else:
            convert_to_mc_dropout(layer)


def activate_mc_dropout(model: nn.Module, activate: bool = True, random: bool = False, verbose: bool = False):
    for layer in model.children():
        if isinstance(layer, DropoutMC):
            if verbose:
                print(f"Current DO state: {layer.activate}")
                print(f"Switching state to: {activate}")
            layer.activate = activate
            if activate and random:
                layer.p = torch.clamp(torch.rand(1), 0, 0.5).item()
            if not activate and random:
                layer.p = layer.p_init
        else:
            activate_mc_dropout(layer, activate=activate, random=random, verbose=verbose)
