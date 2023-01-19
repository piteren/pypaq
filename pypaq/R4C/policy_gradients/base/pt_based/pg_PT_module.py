"""

 2022 (c) piteren

    baseline PG Module for MOTorch (PyTorch)

"""

import torch

from pypaq.torchness.motorch import Module
from pypaq.torchness.layers import LayDense, zeroes
from pypaq.torchness.base_elements import scaled_cross_entropy


class PGModel(Module):

    def __init__(
            self,
            observation_width=  4,
            num_actions: int=   2,
            hidden_layers=      (20,),
            lay_norm=           False,
            use_scaled_ce=      True,  # experimental Scaled Cross Entropy loss
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

        self.use_scaled_ce = use_scaled_ce

    def forward(self, observation) -> dict:

        out = self.ln(observation) if self.lay_norm else observation

        zsL = []
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            zsL.append(zeroes(out))
            if self.lay_norm: out = ln(out)

        logits = self.logits(out)
        probs = torch.nn.functional.softmax(input=logits, dim=-1)

        return {
            'logits':       logits,
            'probs':        probs,
            'zeroes':       zsL}

    def loss(self, observation, action_taken, dreturn) -> dict:

        out = self(observation)
        logits = out['logits']

        if self.use_scaled_ce:
            actor_ce_scaled = scaled_cross_entropy(
                labels= action_taken,
                scale=  dreturn,
                probs=  out['probs'])
        else:
            actor_ce = torch.nn.functional.cross_entropy(logits, action_taken, reduction='none')
            actor_ce_scaled = actor_ce * dreturn

        out.update({'loss': torch.mean(actor_ce_scaled)})

        return out