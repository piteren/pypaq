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
            feats,
            to_torch=   True,
            to_devices= True) -> DTNS:
        _ , kwargs = self._conv_move(feats=feats, to_torch=to_torch, to_devices=to_devices)
        return self._nngraph_module.reduce(**kwargs)

    def reduce_pyramidal(
            self,
            feats,
            to_torch=   True,
            to_devices= True,
            **kwargs) -> DTNS:
        _ , kw = self._conv_move(feats=feats, to_torch=to_torch, to_devices=to_devices)
        return self._nngraph_module.reduce_pyramidal(**kw, **kwargs)


def get_FeatsSeqReductor_MOTorch(**kwargs) -> FeatsSeqReductor_MOTorch:
    return FeatsSeqReductor_MOTorch(
        nngraph=    FeatsSeqReductor,
        **kwargs)