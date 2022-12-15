from typing import Optional
import torch

from pypaq.torchness.types import ACT, INI, TNS, DTNS
from pypaq.torchness.base_elements import my_initializer
from pypaq.torchness.layers import LayDense, zeroes


class LayDRT(torch.nn.Module):

    def __init__(
            self,
            in_width: int,
            do_scaled_dns: bool=    False,          # two denses (True) or single dense (False)
            dns_scale: int=         4,              # up-scale for first dense of two
            activation: ACT=        torch.nn.ReLU,
            lay_dropout: float=     0.0,            # dropout after dense/s
            residual: bool=         True,           # residual yes/no
            res_dropout: float=     0.0,            # dropout on residual connection
            device=                 None,
            dtype=                  None,
            initializer: INI=       None):

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
            # just single dense, with activation
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

    def forward(self, input:TNS) -> DTNS:

        out = self.ln_in(input)

        dense = self.denses[0]
        out = dense(out)
        zsL = [zeroes(out)]

        if len(self.denses) > 1: # there is second one, without activation
            dense = self.denses[1]
            out = dense(out)

        if self.drop_lay:
            out = self.drop_lay(out)

        if self.add_res:
            if self.drop_res: x = self.drop_res(input)
            out += input # residual

        return {
            'out':  out,
            'zsL':  zsL}

# Deep Residual encoder based on stacked LayDRT
class EncDRT(torch.nn.Module):

    def __init__(
            self,
            in_width: int,
            in_dropout: float=          0.0,            # dropout on input
            shared_lays: bool=          False,          # shared variables in enc_layers
            n_layers: int=              6,
            lay_width: Optional[int]=   None,           # for None matches input width
            do_scaled_dns: bool=        True,
            dns_scale: int=             4,              # scale(*) of first dense
            activation: ACT=            torch.nn.ReLU,  # gelu is really worth a try
            lay_dropout: float=         0.0,            # dropout after two denses
            residual: bool=             True,           # residual yes/no
            res_dropout: float=         0.0,            # dropout on residual connection
            device=                     None,
            dtype=                      None,
            initializer: INI=           None):

        super(EncDRT, self).__init__()

        if initializer is None: initializer = my_initializer

        self.in_drop_lay = torch.nn.Dropout(p=in_dropout) if in_dropout else None

        self.in_width = in_width
        self.lay_width = lay_width or self.in_width
        self.projection_lay = LayDense(
            in_features=    self.in_width,
            out_features=   self.lay_width,
            activation=     None,
            bias=           False,
            device=         device,
            dtype=          dtype,
            initializer=    initializer) if self.lay_width != self.in_width else None

        self.ln_in = torch.nn.LayerNorm(
            normalized_shape=   self.lay_width,
            device=             device,
            dtype=              dtype)

        num_layers_to_build = 1 if shared_lays else n_layers
        self.drt_lays = [LayDRT(
            in_width=       self.lay_width,
            do_scaled_dns=  do_scaled_dns,
            dns_scale=      dns_scale,
            activation=     activation,
            lay_dropout=    lay_dropout,
            residual=       residual,
            res_dropout=    res_dropout,
            device=         device,
            dtype=          dtype,
            initializer=    initializer) for _ in range(num_layers_to_build)]
        for lix,lay in enumerate(self.drt_lays): self.add_module(f'lay_drt_{lix}',lay)
        if shared_lays and n_layers > 1: self.drt_lays *= n_layers

    def forward(self, input:TNS) -> DTNS:

        zsL = []

        out = input

        if self.in_drop_lay: # input dropout
            out = self.in_drop_lay(out)

        if self.projection_lay: # input projection, no activation <- do not catch zeroes
            out = self.projection_lay(out)

        out = self.ln_in(out)

        for drt_lay in self.drt_lays:
            lay_out = drt_lay(out)
            out = lay_out['out']
            zsL += lay_out['zsL']

        return {
            'out':  out,
            'zsL':  zsL}