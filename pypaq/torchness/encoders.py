from typing import Callable, Optional
import torch

from pypaq.torchness.base_elements import my_initializer
from pypaq.torchness.layers import LayDense, zeroes


class LayDRT(torch.nn.Module):

    def __init__(
            self,
            in_width: int,
            do_scaled_dns=  False,          # two denses (True) or single dense (False)
            dns_scale=      4,              # scale for two denses
            activation=     torch.nn.ReLU,
            lay_dropout=    0.0,            # dropout after dense/s
            add_res=        True,           # residual yes/no
            res_dropout=    0.0,            # dropout on residual connection
            training_flag=                      None,           # training flag tensor (for dropout)
            device=                             None,
            dtype=                              None,
            initializer: Optional[Callable]=    None):

        torch.nn.Module.__init__(self)

        if initializer is None: initializer = my_initializer

        self.in_width = in_width

        self.ln_in = torch.nn.LayerNorm(self.in_width)

        self.denses = []
        if do_scaled_dns:
            # dense (scale up) with activation
            self.denses.append(LayDense(
                in_features=    self.in_width,
                out_features=   self.in_width * dns_scale,
                activation=     activation,
                bias=           True,
                device=         device,
                dtype=          dtype,
                initializer=    initializer))
            # dense (scale down) without activation
            self.denses.append(LayDense(
                in_features=    self.in_width * dns_scale,
                out_features=   self.in_width,
                activation=     None,
                bias=           True,
                device=         device,
                dtype=          dtype,
                initializer=    initializer))
        else:
            # just dense
            self.denses.append(LayDense(
                in_features=    self.in_width,
                out_features=   self.in_width,
                activation=     activation,
                bias=           True,
                device=         device,
                dtype=          dtype,
                initializer=    initializer))

        raise NotImplemented
        #initializer(self.dense.weight)
        #torch.nn.init.zeros_(self.dense.bias)

    def forward(self, x):

        zsL = []

        out = self.ln_in(x)


        print(out)

        return {
            'out':  out,
            'zsL':  zsL}

# my dense layer, adds initializer and activation
class EncDRT(torch.nn.Module):

    def __init__(
            self,
            initializer: Optional[Callable] = None):

        torch.nn.Module.__init__(self)
        raise NotImplemented
        pass

    def forward(self, x):
        pass