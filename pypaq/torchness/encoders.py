from typing import Callable, Optional
import torch

from pypaq.torchness.base_elements import my_initializer
from pypaq.torchness.layers import LayDense, zeroes


class LayDRT(torch.nn.Module):

    def __init__(
            self,
            in_width: int,
            do_scaled_dns: bool=                            False,          # two denses (True) or single dense (False)
            dns_scale: int=                                 4,              # scale for two denses
            activation: Optional[type(torch.nn.Module)]=    torch.nn.ReLU,
            lay_dropout: float=                             0.0,            # dropout after dense/s
            residual: bool=                                 True,           # residual yes/no
            res_dropout: float=                             0.0,            # dropout on residual connection
            device=                                         None,
            dtype=                                          None,
            initializer: Optional[Callable]=                None):

        #TODO: check module devices & dtype

        super(LayDRT, self).__init__()

        if initializer is None: initializer = my_initializer

        self.ln_in = torch.nn.LayerNorm(
            normalized_shape=   in_width,
            device=             device,
            dtype=              dtype)

        self.denses = []
        if do_scaled_dns:
            # dense (scale up) with activation
            self.denses.append(LayDense(
                in_features=    in_width,
                out_features=   in_width * dns_scale,
                activation=     activation,
                bias=           True,
                device=         device,
                dtype=          dtype,
                initializer=    initializer))
            # dense (scale down) without activation
            self.denses.append(LayDense(
                in_features=    in_width * dns_scale,
                out_features=   in_width,
                activation=     None,
                bias=           True,
                device=         device,
                dtype=          dtype,
                initializer=    initializer))
        else:
            # just dense
            self.denses.append(LayDense(
                in_features=    in_width,
                out_features=   in_width,
                activation=     activation,
                bias=           True,
                device=         device,
                dtype=          dtype,
                initializer=    initializer))
        for dix, l in enumerate(self.denses): self.add_module(f'dense{dix}', l)

        self.drop_lay = torch.nn.Dropout(p=lay_dropout) if lay_dropout else None

        self.add_res = residual

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