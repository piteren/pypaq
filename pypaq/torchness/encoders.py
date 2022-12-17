from typing import Optional
import torch

from pypaq.torchness.types import ACT, INI, TNS, DTNS
from pypaq.torchness.base_elements import my_initializer
from pypaq.torchness.layers import LayDense, TF_Dropout, LayConv1D, zeroes


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
            initializer=    initializer
        ) for _ in range(num_layers_to_build)]
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

# CNN 1D Encoder (for sequences, LN > CNN > act > drop > RES), number of parameters: n_layers*kernel*in_features*n_filters
class EncCNN(torch.nn.Module):

    def __init__(
            self,
            in_features: int,                           # input num of channels
            time_drop: float=           0.0,
            feat_drop: float=           0.0,
            # layer
            shared_lays: bool=          False,          # shared variables in enc_layers
            n_layers :int=              6,              # num of layers
            kernel_size :int=           3,              # layer kernel
            n_filters :int=             32,             # num of filters
            activation: ACT=            torch.nn.ReLU,  # global enc activation func
            lay_dropout: float=         0.0,
            # lay_DRT
            do_ldrt=                    False,          # lay DRT - build or not
            ldrt_do_scaled_dns: bool=   True,
            ldrt_dns_scale: int=        4,
            ldrt_drop: float or None=   0.0,
            ldrt_residual: bool=        True,
            ldrt_res_dropout: float=    0.0,
            # other
            device=                     None,
            dtype=                      None,
            initializer: INI=           None):

        super(EncCNN, self).__init__()

        if initializer is None: initializer = my_initializer

        self.in_features = in_features
        self.n_filters = n_filters

        self.in_TFdrop_lay = TF_Dropout(
            time_drop=  time_drop,
            feat_drop=  feat_drop) if time_drop or feat_drop else None

        self.projection_lay = LayDense(
            in_features=    self.in_features,
            out_features=   self.n_filters,
            activation=     None,
            bias=           False,
            device=         device,
            dtype=          dtype,
            initializer=    initializer) if self.in_features != self.n_filters else None

        num_layers_to_build = 1 if shared_lays else n_layers

        self.lay_lnL = [torch.nn.LayerNorm(
            normalized_shape=   self.n_filters,
            device=             device,
            dtype=              dtype
        ) for _ in range(num_layers_to_build)]
        for lix,lay in enumerate(self.lay_lnL): self.add_module(f'lay_ln_{lix}', lay)

        self.lay_conv1DL = [LayConv1D(
            in_features=    self.n_filters,
            n_filters=      n_filters,
            kernel_size=    kernel_size,
            device=         device,
            dtype=          dtype,
            activation=     None,
            initializer=    initializer
        ) for _ in range(num_layers_to_build)]
        for lix,lay in enumerate(self.lay_conv1DL): self.add_module(f'lay_conv_{lix}', lay)

        self.lay_dropL = [torch.nn.Dropout(
            p=  lay_dropout
        ) if lay_dropout else None for _ in range(num_layers_to_build)]
        if lay_dropout:
            for lix,lay in enumerate(self.lay_dropL): self.add_module(f'lay_drop_{lix}', lay)

        self.lay_DRTL = [LayDRT(
            in_width=       self.n_filters,
            do_scaled_dns=  ldrt_do_scaled_dns,
            dns_scale=      ldrt_dns_scale,
            activation=     activation,
            lay_dropout=    ldrt_drop,
            residual=       ldrt_residual,
            res_dropout=    ldrt_res_dropout,
            device=         device,
            dtype=          dtype,
            initializer=    initializer
        ) if do_ldrt else None for _ in range(num_layers_to_build)]
        if do_ldrt:
            for lix,lay in enumerate(self.lay_DRTL): self.add_module(f'lay_DRT_{lix}', lay)

        if shared_lays and n_layers > 1:
            self.lay_lnL *= n_layers
            self.lay_conv1DL *= n_layers
            self.lay_dropL *= n_layers
            self.lay_DRTL *= n_layers

        self.activation = activation() if activation else None

        self.out_ln = torch.nn.LayerNorm(
            normalized_shape=   self.n_filters,
            device=             device,
            dtype=              dtype)

    def forward(self, input:TNS, history:Optional[TNS]=None) -> DTNS:

        input_lays = []  # here we will store inputs of the following layers to extract the state (history)
        zsL = []

        if self.in_TFdrop_lay:
            input = self.in_TFdrop_lay(input)

        if self.projection_lay:
            input = self.projection_lay(input)

        output = input      # for 0 layers case
        sub_input = input   # first input
        #for lay_ln, lay_conv1D, lay_drop, lay_DRT in zip(self.lay_lnL, self.lay_conv1DL, self.lay_dropL, self.lay_DRTL):
        for lix in range(len(self.lay_conv1DL)):

            lay_ln = self.lay_lnL[lix]
            lay_conv1D = self.lay_conv1DL[lix]
            lay_drop = self.lay_dropL[lix]
            lay_DRT = self.lay_DRTL[lix]

            # TODO: concat with history
            lay_input = sub_input
            input_lays.append(lay_input)

            lay_input = lay_ln(lay_input)
            print('applied ln')

            output = lay_conv1D(lay_input)
            print('applied conv')

            if self.activation:
                output = self.activation(output)
                zsL += zeroes(output)

            if lay_drop:
                output = lay_drop(output)

            output += sub_input # RES # TODO: maybe add here res_dropout to sub_input like in LayDRT

            if lay_DRT:
                lay_out = lay_DRT(output)
                output = lay_out['output']
                zsL += lay_out['zeroes']

            sub_input = output

        output = self.out_ln(output)

        # TODO: prepare fin_state
        fin_state = None
        """
        if history is not None:
            state = tf.stack(input_lays, axis=-3)
            if verb > 1: print(f' > state (stacked): {state}')
            fin_state = tf.split(state, num_or_size_splits=[-1, kernel - 1], axis=-2)[1]
            if verb > 1: print(f' > fin_state (split): {fin_state}')
        """

        return {
            'out':      output,
            'state':    fin_state, # history for next
            'zsL':      zsL}