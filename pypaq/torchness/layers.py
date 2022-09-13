from typing import Callable, Optional
import torch

from pypaq.torchness.base_elements import my_initializer


# my dense layer, adds initializer
class LayDense(torch.nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool=                         True,
            device=                             None,
            dtype=                              None,
            initializer: Optional[Callable]=    None):

        super(LayDense, self).__init__()
        self.dense = torch.nn.Linear(
            in_features=    in_features,
            out_features=   out_features,
            bias=           bias,
            device=         device,
            dtype=          dtype)

        if initializer is None: initializer = my_initializer
        initializer(self.dense.weight)
        if bias: torch.nn.init.zeros_(self.dense.bias)

    def forward(self, x):
        return self.dense(x)

# time & feats dropout (for sequences)
class TF_Dropout(torch.nn.Module):

    def __init__(
            self,
            time_drop: float,
            feat_drop: float):

        super(TF_Dropout, self).__init__()
        self.tdl = torch.nn.Dropout(p=time_drop) if time_drop else None
        self.fdl = torch.nn.Dropout(p=feat_drop) if feat_drop else None

    def forward(
            self,
            x: torch.Tensor # tensor [batch,seq,feats]
    ) -> torch.Tensor:

        output = x
        in_shape = x.size()

        if self.tdl is not None:
            t_drop = torch.ones(in_shape[-2])
            t_drop = self.tdl(t_drop)
            t_drop = torch.unsqueeze(t_drop, dim=-1)
            output = output * t_drop

        if self.fdl is not None:
            f_drop = torch.ones(in_shape[-1])
            f_drop = self.fdl(f_drop)
            f_drop = torch.unsqueeze(f_drop, dim=-2)
            output = output * f_drop

        return output