import math
import torch

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

            #torch.nn.init.zeros_(self.bias) # my old way

            # original Linear reset for bias
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input:TNS) -> TNS:
        out = super().forward(input)
        if self.activation: out = self.activation(out)
        return out

# time & feats dropout (for sequences), input tensor [...,seq,feats]
class TF_Dropout(torch.nn.Dropout):

    def __init__(
            self,
            time_drop: float=   0.0,
            feat_drop: float=   0.0,
            **kwargs):
        self.time_drop = time_drop
        self.feat_drop = feat_drop
        super(TF_Dropout, self).__init__(**kwargs)

    def forward(self, input:TNS) -> TNS:

        output = input
        in_shape = input.size()

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

# my Conv1d, with initializer + activation
class LayConv1D(torch.nn.Conv1d):

    def __init__(
            self,
            in_features: int,                                               # input num of channels
            n_filters: int,                                                 # output num of channels
            kernel_size=        3,
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
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input:TNS) -> TNS:
        inp_trans = torch.transpose(input=input, dim0=-1, dim1=-2) # transposes input to (N,C,L) <- (N,L,C), since torch.nn.Conv1d assumes that channels is @ -2 dim
        out = super().forward(input=inp_trans)
        out = torch.transpose(out, dim0=-1, dim1=-2) # transpose back
        if self.activation: out = self.activation(out)
        return out

"""
# TODO: To Be Implemented
class Attn(torch.nn.Module):

    def __init__(self):
        super(Attn, self).__init__()
"""

# returns [0,1] tensor: 1 where input not activated (value =< 0), looks at last dimension / features
def zeroes(input :TNS) -> TNS:
    axes = [ix for ix in range(len(input.shape))][:-1]  # all but last(feats) axes indexes list like: [0,1,2] for 4d shape
    activated = torch.where(                            # 1 for value greater than zero, other 0
        condition=      torch.gt(input, 0),
        input=          torch.ones_like(input),         # true
        other=          torch.zeros_like(input))        # false
    activated_reduced = torch.sum(activated, dim=axes) if axes else activated  # 1 or more for activated, 0 for not activated, if not axes -> we have only-feats-tensor-case
    not_activated = torch.eq(activated_reduced, 0)      # true where summed gives zero (~invert)
    return not_activated.to(dtype=torch.int8)           # cast to int