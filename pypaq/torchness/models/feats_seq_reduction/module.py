import torch

from pypaq.torchness.motorch import Module
from pypaq.torchness.types import ACT, TNS, DTNS
from pypaq.torchness.encoders import EncTNS



# Features Sequence Reductor based on TAT
class FeatsSeqReductor(Module):

    def __init__(
            self,
            num_layers_TAT: int,
            d_model: int,
            shared_lays=                            None,
            max_seq_len=                            None,
            nhead: int=                             8,
            dns_scale: int=                         4,
            dropout: float=                         0.1,
            dropout_att: float=                     0.0,
            activation: ACT=                        torch.nn.ReLU,
            dropout_res: float=                     0.0,
            device=                                 None,
            dtype=                                  None):

        Module.__init__(self)

        self.enc = EncTNS(
            num_layers=     0,
            num_layers_TAT= num_layers_TAT,
            shared_lays=    shared_lays,
            max_seq_len=    max_seq_len,
            d_model=        d_model,
            nhead=          nhead,
            dns_scale=      dns_scale,
            dropout=        dropout,
            dropout_att=    dropout_att,
            activation=     activation,
            dropout_res=    dropout_res,
            device=         device,
            dtype=          dtype)

    def reduce(self, feats: TNS) -> DTNS:
        return self.enc(feats)

    def forward(self, *args, **kwargs) -> DTNS:
        raise NotImplementedError

    def loss_acc(self, *args, **kwargs) -> DTNS:
        raise NotImplementedError