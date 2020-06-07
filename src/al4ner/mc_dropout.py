from typing import Tuple

import torch
from torch import nn
import flair

class DropoutMC(nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x):
        return nn.functional.dropout(x, self.p, training=self.training or self.activate)

class LockedDropoutMC(torch.nn.Module):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, dropout_rate=0.5, batch_first=True, inplace=False, activate=False):
        super(LockedDropoutMC, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout_rate_init = dropout_rate
        self.batch_first = batch_first
        self.inplace = inplace
        self.activate = activate

    def forward(self, x):
        if self.training:
            self.activate = True
        #if not self.training or not self.dropout_rate:
        if not self.activate or not self.dropout_rate:
            return x

        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)

class WordDropoutMC(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def __init__(self, dropout_rate=0.05, inplace=False, activate=False):
        super(WordDropoutMC, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout_rate_init = dropout_rate
        self.inplace = inplace
        self.activate = activate

    def forward(self, x):
        if self.training:
            self.activate = True

        #if not self.training or not self.dropout_rate:
        if not self.activate or not self.dropout_rate:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


def convert_to_mc_dropout(model: nn.Module, layers_to_substitute: Tuple[nn.Module] = (nn.Dropout,), option='torch'):
    if option == 'flair':
        for i, layer in enumerate(list(model.children())):
            if isinstance(layer, flair.nn.LockedDropout):
                name = list(model._modules.items())[i][0]
                model._modules[name] = LockedDropoutMC(dropout_rate=layer.dropout_rate, activate=True)
            elif isinstance(layer, flair.nn.WordDropout):
                name = list(model._modules.items())[i][0]
                model._modules[name] = WordDropoutMC(dropout_rate=layer.dropout_rate, activate=True)
            else:
                convert_to_mc_dropout(layer)

    elif option == 'torch':
        for i, layer in enumerate(list(model.children())):
            if isinstance(layer, layers_to_substitute):
                name = list(model._modules.items())[i][0]
                model._modules[name] = DropoutMC(p=layer.p, activate=True)
            else:
                convert_to_mc_dropout(layer)


def activate_mc_dropout(model: nn.Module, activate: bool = True, if_custom_rate: bool = False,
                        custom_rate: float = 0.25, random: bool = False,
                        layers_to_activate: Tuple[nn.Module] = (WordDropoutMC, LockedDropoutMC),
                        verbose: bool = False, option='torch'):

    if option == 'flair':
        for layer in model.children():
            if isinstance(layer, layers_to_activate):
                if verbose:
                    print(f"Current DO state: {layer.activate}")
                    print(f"Switching state to: {activate}")
                layer.activate = activate
                if activate and if_custom_rate:
                    print('CUSTOM RATE: ', custom_rate)
                    layer.dropout_rate = custom_rate
                if not activate and if_custom_rate:
                    layer.dropout_rate = layer.dropout_rate_init

            else:
                activate_mc_dropout(layer, activate=activate, random=random, verbose=verbose)

    elif option == 'torch':
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

