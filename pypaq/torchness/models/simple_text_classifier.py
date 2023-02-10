import math
import numpy as np
from typing import List, Optional, Union
from tqdm import tqdm

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.torchness.types import TNS, DTNS, INI
from pypaq.torchness.motorch import Module, MOTorch
from pypaq.torchness.models.text_embbeder import TextEMB
from pypaq.torchness.models.simple_feats_classifier import SFeatsCSF



# Simple Text Classification Module, based on Sentence-Transformer
class STextCSF(Module):

    def __init__(
            self,
            st_name: str=       'all-MiniLM-L6-v2',
            enc_batch_size=                         256,

            in_drop: float=                         0.0,
            mid_width: int=                         30,
            mid_drop: float=                        0.0,
            num_classes: int=                       2,
            class_weights: Optional[List[float]]=   None,
            initializer: INI=                       None,
            dtype=                                  None,
            logger=                                 None):

        if not logger: logger = get_pylogger()
        self.logger = logger

        Module.__init__(self)

        self.te_module = TextEMB(
            st_name=        st_name,
            enc_batch_size= enc_batch_size)

        self.csf = SFeatsCSF(
            feats_width=    self.te_module.width,
            in_drop=        in_drop,
            mid_width=      mid_width,
            mid_drop=       mid_drop,
            num_classes=    num_classes,
            class_weights=  class_weights,
            initializer=    initializer,
            dtype=          dtype,
            logger=         self.logger)

    def encode(
            self,
            texts: Union[str, List[str]],
            show_progress_bar=  'auto',
            device=             None) -> DTNS:
        if show_progress_bar == 'auto':
            show_progress_bar = False
            if type(texts) is list and len(texts) > 1000:
                show_progress_bar = True
        embeddings = self.te_module.encode(
            texts=              texts,
            show_progress_bar=  show_progress_bar,
            device=             device)
        return {'embeddings': embeddings}

    def forward(self, feats:TNS) -> DTNS:
        return self.csf(feats)

    def loss(self, feats:TNS, labels:TNS) -> DTNS:
        return self.csf.loss(feats, labels)


class STextCSF_MOTorch(MOTorch):

    def __init__(
            self,
            module_type: Optional[type(STextCSF)]=  STextCSF,
            enc_batch_size=                         128,    # number of lines in batch for embeddings
            fwd_batch_size=                         256,    # number of embeddings in batch for probs
            **kwargs):

        MOTorch.__init__(
            self,
            module_type=    module_type,
            enc_batch_size= enc_batch_size,
            fwd_batch_size= fwd_batch_size,
            **kwargs)

    def get_embeddings(
            self,
            lines: Union[List[str],str],
            show_progress_bar=      'auto') -> np.ndarray:
        if type(lines) is str: lines = [lines]
        self.logger.info(f'{self.name} prepares embeddings for {len(lines)} lines..')
        if show_progress_bar == 'auto':
            show_progress_bar = self.logger.level < 21 and len(lines) > 1000
        out = self.module.encode(
            texts=              lines,
            show_progress_bar=  show_progress_bar,
            device=             self.device) # needs to give device here because of SentenceTransformer bug in encode() #153
        return out['embeddings']

    def get_probs(self, lines:List[str]) -> np.ndarray:

        embs = self.get_embeddings(lines)

        num_splits = math.ceil(embs.shape[0] / self['fwd_batch_size']) # INFO: gives +- batch_size
        featsL = np.array_split(embs,num_splits)

        self.logger.info(f'{self.name} computes probs for {len(featsL)} batches of embeddings')
        iter = tqdm(featsL) if self.logger.level < 21 else featsL
        probsL = [self(feats)['probs'] for feats in iter]
        probs = np.concatenate(probsL)
        self.logger.info(f'> got probs {probs.shape}')

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
