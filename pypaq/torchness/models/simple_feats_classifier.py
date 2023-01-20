import torch
from typing import List, Optional

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.torchness.motorch import Module
from pypaq.torchness.types import TNS, DTNS, INI
from pypaq.torchness.base_elements import my_initializer
from pypaq.torchness.layers import LayDense


# Simple Feats Classification Module
class SFeatsCSF(Module):

    def __init__(
            self,
            feats_width: int,
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

        if initializer is None: initializer = my_initializer

        self.drop = torch.nn.Dropout(p=in_drop) if in_drop else None

        self.logger.info(f'SFeatsCSF Module inits for feats of width {feats_width}')
        self.mid = LayDense(
            in_features=    feats_width,
            out_features=   mid_width,
            activation=     torch.nn.ReLU,
            bias=           True,
            initializer=    initializer,
            dtype=          dtype)

        self.mid_drop = torch.nn.Dropout(p=mid_drop) if mid_drop else None

        self.logits = LayDense(
            in_features=    mid_width,
            out_features=   num_classes,
            activation=     None,
            bias=           False,
            initializer=    initializer,
            dtype=          dtype)

        if class_weights:
            class_weights = torch.nn.Parameter(torch.tensor(class_weights), requires_grad=False)
        self.class_weights = class_weights

    def forward(self, feats:TNS) -> DTNS:
        if self.drop: feats = self.drop(feats)
        mid = self.mid(feats)
        if self.mid_drop: mid = self.mid_drop(mid)
        logits = self.logits(mid)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        return {
            'logits':   logits,
            'probs':    probs.detach().cpu().numpy(),
            'preds':    preds.detach().cpu().numpy()}

    def loss(self, feats:TNS, labels:TNS) -> DTNS:

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