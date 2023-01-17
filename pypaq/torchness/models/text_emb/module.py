import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

from pypaq.torchness.motorch import Module, MOTorchException
from pypaq.torchness.types import DTNS


class TextEMBModule(Module):

    def __init__(
            self,
            st_name: str,
            enc_batch_size= 256):
        Module.__init__(self)
        self.st_name = st_name
        self.st_model = SentenceTransformer(model_name_or_path= st_name)
        self.enc_batch_size = enc_batch_size

    def tokenize(self, texts:List[str]) -> List[List[str]]:
        tokenizer = self.st_model.tokenizer
        return [tokenizer.tokenize(t) for t in texts]

    def encode(
            self,
            texts: List[str],
            device=             None,
            show_progress_bar=  True,
    ) -> np.ndarray:
        return self.st_model.encode(
            sentences=          texts,
            batch_size=         self.enc_batch_size,
            show_progress_bar=  show_progress_bar,
            device=             device)

    def forward(self, *args, **kwargs) -> DTNS:
        raise NotImplementedError

    def loss_acc(self, *args, **kwargs) -> DTNS:
        raise NotImplementedError

    @property
    def width(self) -> int:
        return self.st_model.get_sentence_embedding_dimension()