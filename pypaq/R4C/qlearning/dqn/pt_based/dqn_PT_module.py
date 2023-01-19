"""

 2022 (c) piteren

    baseline DQN Module for MOTorch (PyTorch)

"""

import torch
from typing import Dict

from pypaq.torchness.motorch import Module
from pypaq.torchness.layers import LayDense


class DQNModel(Module):

    def __init__(
            self,
            num_actions: int=   4,
            observation_width=  4,
            hidden_layers=      (12,),
            seed=               121):

        torch.nn.Module.__init__(self)

        lay_shapeL = []
        next_in = observation_width
        for hl in hidden_layers:
            lay_shapeL.append((next_in,hl))
            next_in = hl

        self.ln = torch.nn.LayerNorm(observation_width) # input layer norm

        self.linL = [LayDense(*shape) for shape in lay_shapeL]
        self.lnL = [torch.nn.LayerNorm(shape[-1]) for shape in lay_shapeL]

        lix = 0
        for lin,ln in zip(self.linL, self.lnL):
            self.add_module(f'lay_lin{lix}', lin)
            self.add_module(f'lay_ln{lix}', ln)
            lix += 1

        self.logits = LayDense(
            in_features=    next_in,
            out_features=   num_actions,
            activation=     None)

        self.loss = torch.nn.MSELoss(reduction='none')

    def forward(self, obs) -> dict:
        out = self.ln(obs)
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            out = ln(out)
        logits = self.logits(out)
        return {'logits': logits}

    def accuracy(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor) -> float:
        # no good accuracy for this model
        return 0.0

    def loss(self, obs, lbl, mask=None) -> dict:
        out = self(obs)
        logits = out['logits']
        loss = self.loss(logits, lbl)
        if mask is not None: loss *= mask       # mask
        loss = torch.sum(loss, dim=-1)          # reduce over samples
        out['loss'] = torch.mean(loss)          # average
        out['acc'] = self.accuracy(logits, lbl)
        return out