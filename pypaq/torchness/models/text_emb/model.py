"""
    Text Embedding Model
        - prepares embedding for given text (str)
"""

import numpy as np
from typing import List

from pypaq.torchness.models.text_emb.module import TextEMBModule
from pypaq.torchness.motorch import MOTorch


# is MOTorch for given st_name (SentenceTransformer)
class TextEMB_MOTorch(MOTorch):

    def __init__(self, st_name:str, **kwargs):
        name = f'TextedMOTorch_TexEmbMod_{st_name}'
        MOTorch.__init__(self, name=name, st_name=st_name, **kwargs)

    def get_tokens(self, lines: List[str]):
        self._nwwlog.info(f'TextedMOTorch prepares tokens for {len(lines)} lines..')
        return self._nngraph_module.tokenize(lines)

    def get_embeddings(self, lines: List[str]) -> np.ndarray:
        self._nwwlog.info(f'TextedMOTorch prepares embeddings for {len(lines)} lines..')
        return self._nngraph_module.encode(lines)

    @property
    def width(self) -> int:
        return self._nngraph_module.width


def get_TextEMB_MOTorch(st_name:str='all-MiniLM-L6-v2', **kwargs) -> TextEMB_MOTorch:
    return TextEMB_MOTorch(
        nngraph=    TextEMBModule,
        st_name=    st_name,
        **kwargs)