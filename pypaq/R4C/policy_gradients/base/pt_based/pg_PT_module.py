"""

 2022 (c) piteren

    baseline PG Module for MOTorch (PyTorch)

"""

import torch

from pypaq.torchness.motorch import Module
from pypaq.torchness.layers import LayDense, zeroes


class PGModel(Module):

    def __init__(
            self,
            observation_width=  4,
            num_actions: int=   2,
            hidden_layers=      (20,),
            lay_norm=           False,
            seed=               121):

        torch.nn.Module.__init__(self)

        lay_shapeL = []
        next_in = observation_width
        for hl in hidden_layers:
            lay_shapeL.append((next_in,hl))
            next_in = hl

        self.lay_norm = lay_norm

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

    def forward(self, obs) -> dict:

        out = self.ln(obs) if self.lay_norm else obs

        zsL = []
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            zsL.append(zeroes(out))
            if self.lay_norm: out = ln(out)

        logits = self.logits(out)
        probs = torch.nn.functional.softmax(input=logits, dim=-1)

        max_probs = torch.max(probs, dim=-1)  # max action_probs
        min_probs = torch.min(probs, dim=-1)  # min action_probs
        amax_prob = torch.mean(max_probs[0])  # average of batch max action_prob
        amin_prob = torch.mean(min_probs[0])  # average of batch min action_prob

        return {
            'logits':       logits,
            'probs':        probs,
            'amax_prob':    amax_prob,
            'amin_prob':    amin_prob,
            'zeroes':       zsL}

    def loss_acc(self, obs, lbl, ret) -> dict:

        out = self(obs)
        logits = out['logits']

        actor_ce = torch.nn.functional.cross_entropy(logits, lbl, reduction='none')

        out['actor_ce_mean'] = torch.mean(actor_ce)
        out['loss'] = torch.mean(ret * actor_ce)
        out['acc'] = self.accuracy(logits, lbl) # using baseline

        return out