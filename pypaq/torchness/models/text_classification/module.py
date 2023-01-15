import torch
from typing import List, Optional, Union

from pypaq.torchness.motorch import Module
from pypaq.torchness.types import DTNS, INI
from pypaq.torchness.base_elements import my_initializer
from pypaq.torchness.layers import LayDense
from pypaq.torchness.models.text_emb.module import TextEMBModule



# Text Classification Module based on Sentence-Transformers
class TeXClas(Module):

    def __init__(
            self,
            enc_batch_size=                         256,
            st_name: str=                           'all-MiniLM-L6-v2',
            in_drop: float=                         0.0,
            mid_width: int=                         30,
            mid_drop: float=                        0.0,
            num_classes: int=                       2,
            class_weights: Optional[List[float]]=   None,
            initializer: INI=                       None):

        Module.__init__(self)

        self.te_module = TextEMBModule(
            st_name=        st_name,
            enc_batch_size= enc_batch_size)

        if initializer is None: initializer = my_initializer

        self.drop = torch.nn.Dropout(p=in_drop) if in_drop else None

        self.mid = LayDense(
            in_features=    self.te_module.width,
            out_features=   mid_width,
            activation=     torch.nn.ReLU,
            bias=           True,
            initializer=    initializer)

        self.mid_drop = torch.nn.Dropout(p=mid_drop) if mid_drop else None

        self.logits = LayDense(
            in_features=    mid_width,
            out_features=   num_classes,
            activation=     None,
            bias=           False,
            initializer=    initializer)

        if class_weights:
            class_weights = torch.nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        self.class_weights = class_weights

        self.enc_batch_size = enc_batch_size

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

    def forward(self, feats) -> DTNS:
        if self.drop: feats = self.drop(feats)
        mid = self.mid(feats)
        if self.mid_drop: mid = self.mid_drop(mid)
        logits = self.logits(mid)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        labels = torch.argmax(logits, dim=-1)
        return {
            'logits':   logits,
            'probs':    probs,
            'labels':   labels}

    def loss_acc(self, feats, labels) -> DTNS:
        out = self.forward(feats)
        logits = out['logits']

        loss = torch.nn.functional.cross_entropy(
            input=      logits,
            target=     labels,
            weight=     self.class_weights,
            reduction=  'mean')
        acc = self.accuracy(logits, labels)
        f1 = self.f1(logits, labels)
        out.update({
            'loss': loss,
            'acc':  acc,
            'f1':   f1})
        return out