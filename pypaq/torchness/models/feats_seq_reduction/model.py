"""
    Features Sequence Reduction Model
        - reduces sequence to single vector
"""

from pypaq.torchness.types import DTNS
from pypaq.torchness.models.feats_seq_reduction.module import FeatsSeqReductor
from pypaq.torchness.motorch import MOTorch


# is MOTorch with reduce()
class FeatsSeqReductor_MOTorch(MOTorch):

    def reduce(
            self,
            *args,
            to_torch=   True,
            to_devices= True,
            **kwargs) -> DTNS:
        args, kwargs = self._conv_move(*args, to_torch=to_torch, to_devices=to_devices, **kwargs)
        return self._nngraph_module.reduce(*args, **kwargs)


def get_FeatsSeqReductor_MOTorch(**kwargs) -> FeatsSeqReductor_MOTorch:
    return FeatsSeqReductor_MOTorch(
        nngraph=    FeatsSeqReductor,
        **kwargs)