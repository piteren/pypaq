"""
    Text Classification MOTorch
        - text (str) classifier
"""

import math
import numpy as np
from typing import List, Optional, Union
from tqdm import tqdm

from pypaq.torchness.motorch import MOTorch
from pypaq.torchness.models.text_classification.module import TeXClas



class TeXClas_MOTorch(MOTorch):

    def __init__(
            self,
            nngraph: Optional[type(TeXClas)]=   TeXClas,
            enc_batch_size=                     128,    # number of lines in batch for embeddings
            fwd_batch_size=                     256,    # number of embeddings in batch for probs
            **kwargs):

        MOTorch.__init__(
            self,
            nngraph=        nngraph,
            enc_batch_size= enc_batch_size,
            fwd_batch_size= fwd_batch_size,
            **kwargs)

    def get_embeddings(
            self,
            lines: Union[List[str],str],
            show_progress_bar=      'auto') -> np.ndarray:
        if type(lines) is str: lines = [lines]
        self._nwwlog.info(f'TeXClas_MOTorch prepares embeddings for {len(lines)} lines..')
        if show_progress_bar == 'auto':
            show_progress_bar = self._nwwlog.level < 21 and len(lines) > 1000
        out = self._nngraph_module.encode(
            texts=              lines,
            show_progress_bar=  show_progress_bar,
            device=             self._torch_dev) # need to give device here because of SentenceTransformer bug in encode() #153
        return out['embeddings']

    def get_probs(self, lines:List[str]) -> np.ndarray:

        embs = self.get_embeddings(lines)

        num_splits = math.ceil(embs.shape[0] / self['fwd_batch_size']) # INFO: gives +- batch_size
        featsL = np.array_split(embs,num_splits)

        self._nwwlog.info(f'TeXClas_MOTorch computes probs for {len(featsL)} batches of embeddings')
        iter = tqdm(featsL) if self._nwwlog.level < 21 else featsL
        probsL = [self(feats)['probs'].cpu().detach().numpy() for feats in iter]
        probs = np.concatenate(probsL)
        self._nwwlog.info(f'> got probs {probs.shape}')

        return probs

    def get_probsL(self, linesL:List[List[str]]) -> List[np.ndarray]:

        lines = []
        for l in linesL:
            lines += l

        probs = self.get_probs(lines)

        acc_lengths = []
        acc = 0
        for l in [len(ls) for ls in linesL]:
            acc_lengths.append(l+acc)
            acc += l
        acc_lengths.pop(-1)

        if acc_lengths: return np.split(probs,acc_lengths)
        else:           return [probs]
