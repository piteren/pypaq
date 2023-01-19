import numpy as np
import torch
from typing import Optional

from pypaq.torchness.base_elements import TorchnessException
from pypaq.torchness.types import ACT, INI, TNS
from pypaq.torchness.base_elements import my_initializer


# my dense layer, linear with initializer + activation
class LayDense(torch.nn.Linear):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            activation: ACT=    torch.nn.ReLU,
            bias: bool=         True,
            device=             None,
            dtype=              None,
            initializer: INI=   None):
        self.initializer = initializer or my_initializer
        torch.nn.Linear.__init__(
            self,
            in_features=    in_features,
            out_features=   out_features,
            bias=           bias,
            device=         device,
            dtype=          dtype)
        self.activation = activation() if activation else None

    def reset_parameters(self) -> None:

        self.initializer(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

            ### original Linear (with uniform) reset for bias
            # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inp:TNS) -> TNS:
        out = super().forward(inp)
        if self.activation: out = self.activation(out)
        return out

    def extra_repr(self) -> str:
        act_info = '' if self.activation else ', activation=None'
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}{act_info}'

# time & feats dropout (for sequences), inp tensor [...,seq,feats]
class TF_Dropout(torch.nn.Dropout):

    def __init__(
            self,
            time_drop: float=   0.0,
            feat_drop: float=   0.0,
            inplace: bool=      False):
        self.time_drop = time_drop
        self.feat_drop = feat_drop
        super(TF_Dropout, self).__init__(inplace=inplace)

    def forward(self, inp:TNS) -> TNS:

        output = inp
        in_shape = inp.size()

        if self.time_drop:
            t_drop = torch.ones(in_shape[-2])
            t_drop = torch.nn.functional.dropout(
                input=      t_drop,
                p=          self.time_drop,
                training=   self.training,
                inplace=    self.inplace)
            t_drop = torch.unsqueeze(t_drop, dim=-1)
            output = output * t_drop

        if self.feat_drop:
            f_drop = torch.ones(in_shape[-1])
            f_drop = torch.nn.functional.dropout(
                input=      f_drop,
                p=          self.time_drop,
                training=   self.training,
                inplace=    self.inplace)
            f_drop = torch.unsqueeze(f_drop, dim=-2)
            output = output * f_drop

        return output

    def extra_repr(self) -> str:
        return f'time_drop={self.time_drop}, feat_drop={self.feat_drop}, inplace={self.inplace}'

# my Conv1D, with initializer + activation
class LayConv1D(torch.nn.Conv1d):

    def __init__(
            self,
            in_features: int,                   # input num of channels
            n_filters: int,                     # output num of channels
            kernel_size: int=   3,
            stride=             1,              # single number or a one-element tuple
            padding=            'same',
            dilation=           1,
            groups=             1,
            bias=               True,
            padding_mode=       'zeros',
            device=             None,
            dtype=              None,
            activation: ACT=    torch.nn.ReLU,
            initializer: INI=   None):

        super(LayConv1D, self).__init__(
            in_channels=    in_features,
            out_channels=   n_filters,
            kernel_size=    kernel_size,
            stride=         stride,
            padding=        padding,
            dilation=       dilation,
            groups=         groups,
            bias=           bias,
            padding_mode=   padding_mode,
            device=         device,
            dtype=          dtype)

        self.activation = activation() if activation else None

        if not initializer: initializer = my_initializer
        initializer(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

            # original Conv1D (with uniform) reset for bias
            # fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inp:TNS) -> TNS:
        inp_trans = torch.transpose(input=inp, dim0=-1, dim1=-2) # transposes inp to (N,C,L) <- (N,L,C), since torch.nn.Conv1d assumes that channels is @ -2 dim
        out = super().forward(input=inp_trans)
        out = torch.transpose(out, dim0=-1, dim1=-2) # transpose back
        if self.activation: out = self.activation(out)
        return out

# Residual Layer with dropout for bypass inp
class LayRES(torch.nn.Module):

    def __init__(
            self,
            in_features: Optional[int]= None,
            dropout: float=             0.0):

        if dropout and in_features is None:
            raise TorchnessException('LayRES with dropout needs to know its in_features (int) - cannot be None')

        super(LayRES, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout) if dropout else None

    def forward(self, inp:TNS, bypass:TNS) -> TNS:
        if self.dropout:
            bypass = self.dropout(bypass)
        return inp + bypass


# returns [0,1] tensor: 1 where inp not activated (value =< 0), looks at last dimension / features
def zeroes(inp :TNS) -> np.ndarray:
    axes = [ix for ix in range(len(inp.shape))][:-1]  # all but last(feats) axes indexes list like: [0,1,2] for 4d shape
    activated = torch.where(                            # 1 for value greater than zero, other 0
        condition=      torch.gt(inp, 0),
        input=          torch.ones_like(inp),         # true
        other=          torch.zeros_like(inp))        # false
    activated_reduced = torch.sum(activated, dim=axes) if axes else activated  # 1 or more for activated, 0 for not activated, if not axes -> we have only-feats-tensor-case
    not_activated = torch.eq(activated_reduced, 0)      # true where summed gives zero (~invert)
    not_activated = not_activated.to(dtype=torch.int8)  # cast to int
    return not_activated.detach().cpu().numpy()         # to ndarray