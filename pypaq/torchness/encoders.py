from typing import Callable, Optional
import torch

from pypaq.torchness.base_elements import my_initializer
from pypaq.torchness.layers import LayDense, zeroes


class LayDRT(torch.nn.Module):

    def __init__(
            self,
            in_width: int,
            do_scaled_dns=                      False,          # two denses (True) or single dense (False)
            dns_scale=                          4,              # scale for two denses
            activation=                         torch.nn.ReLU,
            lay_dropout=                        0.0,            # dropout after dense/s
            add_res=                            True,           # residual yes/no
            res_dropout=                        0.0,            # dropout on residual connection
            device=                             None,
            dtype=                              None,
            initializer: Optional[Callable]=    None):

        #TODO: what about: devices, init of other layers?

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

        self.drop_lay = torch.nn.Dropout(p=lay_dropout) if lay_dropout else None

        self.add_res = add_res

        self.drop_res = torch.nn.Dropout(p=res_dropout) if res_dropout else None

    def forward(self, x):

        zsL = []

        out = self.ln_in(x)

        for dense in self.denses:
            out = dense(out)
            zsL.append(zeroes(out))

        if self.drop_lay:
            out = self.drop_lay(out)

        if self.add_res:
            if self.drop_res: x = self.drop_res(x)
            out += x # residual

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