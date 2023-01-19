"""
    Text Embedding Model
        - prepares embedding for given text (str)
"""

import numpy as np
from typing import List

from pypaq.torchness.models.text_emb.module import TextEMB
from pypaq.torchness.motorch import MOTorch


# is MOTorch for given st_name (SentenceTransformer)
class TextEMB_MOTorch(MOTorch):

    def __init__(self, st_name:str, **kwargs):
        name = f'TextEMB_MOTorch_{st_name}'
        MOTorch.__init__(self, name=name, st_name=st_name, **kwargs)

    def get_tokens(self, lines: List[str]):
        self._nwwlog.info(f'{self.name} prepares tokens for {len(lines)} lines..')
        return self._nngraph_module.tokenize(lines)

    def get_embeddings(
            self,
            lines: List[str],
            show_progress_bar=  True) -> np.ndarray:
        self._nwwlog.info(f'{self.name} prepares embeddings for {len(lines)} lines..')
        return self._nngraph_module.encode(
            texts=              lines,
            device=             self._torch_dev, # fixes bug of SentenceTransformers.encode() device placement
            show_progress_bar=  show_progress_bar)

    @property
    def width(self) -> int:
        return self._nngraph_module.width


def get_TextEMB_MOTorch(st_name:str='all-MiniLM-L6-v2', **kwargs) -> TextEMB_MOTorch:
    return TextEMB_MOTorch(
        nngraph=    TextEMB,
        st_name=    st_name,
        **kwargs)