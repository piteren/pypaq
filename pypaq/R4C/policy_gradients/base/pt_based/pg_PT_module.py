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

    def loss_acc(self, observation, action_taken, dreturn) -> dict:

        out = self(observation)
        logits = out['logits']

        if self.use_scaled_ce:

            probs = out['probs']
            prob_action_taken = probs[range(len(action_taken)), action_taken]
            actor_ce = -torch.log(prob_action_taken)
            actor_ce_neg = -torch.log(1-prob_action_taken)

            # merge loss for positive and negative advantage
            actor_ce = torch.where(
                condition=  dreturn > 0,
                input=      actor_ce,
                other=      actor_ce_neg)
        else:
            actor_ce = torch.nn.functional.cross_entropy(logits, action_taken, reduction='none')

        out['actor_ce_mean'] = torch.mean(actor_ce)
        out['loss'] = torch.mean(actor_ce * dreturn)
        out['acc'] = self.accuracy(logits, action_taken) # using baseline

        return out