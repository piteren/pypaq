import torch

from pypaq.torchness.motorch import Module
from pypaq.torchness.types import TNS, DTNS
from pypaq.torchness.layers import LayDense, zeroes


# baseline AC Critic Module
class ACCriticModule(Module):

    def __init__(
            self,
            observation_width=  4,
            gamma=              0.99,  # discount factor (gamma)
            num_actions: int=   2,
            hidden_layers=      (24,24),
            lay_norm=           False,
            seed=               121):

        torch.nn.Module.__init__(self)

        self.gamma = gamma

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

        self.qvs = LayDense(
            in_features=    next_in,
            out_features=   num_actions,
            activation=     None)


    def forward(self, observation:TNS) -> DTNS:

        out = self.ln(observation) if self.lay_norm else observation

        zsL = []
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            zsL.append(zeroes(out))
            if self.lay_norm: out = ln(out)

        qvs = self.qvs(out)

        return {
            'qvs':      qvs,
            'zeroes':   zsL}


    def loss(
            self,
            observation: TNS,
            action_taken_OH: TNS, # one-hot vector of action taken
            next_action_qvs: TNS,
            next_action_probs: TNS,
            reward: TNS
    ) -> DTNS:

        out = self(observation)
        qvs = out['qvs']

        qv = torch.sum(qvs * action_taken_OH, dim=-1)
        next_V = torch.sum(next_action_qvs * next_action_probs, dim=-1) # V(next_s)
        labels = reward + self.gamma * next_V
        diff = labels - qv
        loss = torch.mean(diff * diff) # MSE

        out.update({'loss': loss})
        return out