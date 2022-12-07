import torch

from pypaq.torchness.motorch import Module
from pypaq.torchness.layers import LayDense, zeroes
from pypaq.torchness.base_elements import scaled_cross_entropy


class A2CModel(Module):

    def __init__(
            self,
            observation_width=  4,
            num_actions: int=   2,
            two_towers=         False,  # builds separate towers for Actor & Critic
            num_layers=         1,
            layer_width=        50,
            lay_norm=           False,
            use_scaled_ce=      True,  # experimental Scaled Cross Entropy loss
            use_huber=          False,  # for True uses Huber loss for Critic
            seed=               121):

        torch.nn.Module.__init__(self)

        hidden_layers = tuple([layer_width] * num_layers)

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

        self.linL_tower = [LayDense(*shape) for shape in lay_shapeL] if two_towers else []
        self.lnL_tower = [torch.nn.LayerNorm(shape[-1]) for shape in lay_shapeL] if two_towers else []
        lix = 0
        for lin,ln in zip(self.linL_tower, self.lnL_tower):
            self.add_module(f'lay_lin_tower{lix}', lin)
            self.add_module(f'lay_ln_tower{lix}', ln)
            lix += 1

        self.value = LayDense(
            in_features=    next_in,
            out_features=   1,
            activation=     None)

        self.logits = LayDense(
            in_features=    next_in,
            out_features=   num_actions,
            activation=     None)

        self.use_scaled_ce = use_scaled_ce
        self.use_huber = use_huber

    def forward(self, observation) -> dict:

        inp = self.ln(observation) if self.lay_norm else observation

        zsL = []
        out = inp
        for lin,ln in zip(self.linL,self.lnL):
            out = lin(out)
            zsL.append(zeroes(out))
            if self.lay_norm: out = ln(out)

        out_tower = out
        if self.linL_tower:
            out_tower = inp
            for lin,ln in zip(self.linL_tower,self.lnL_tower):
                out_tower = lin(out_tower)
                zsL.append(zeroes(out_tower))
                if self.lay_norm: out_tower = ln(out_tower)
        value = self.value(out_tower)
        value = torch.reshape(value, (value.shape[:-1])) # remove last dim

        logits = self.logits(out)
        probs = torch.nn.functional.softmax(input=logits, dim=-1)

        return {
            'value':    value,
            'logits':   logits,
            'probs':    probs,
            'zeroes':   zsL}

    def loss_acc(self, observation, action_taken, dreturn) -> dict:

        out = self(observation)
        value = out['value']
        logits = out['logits']

        advantage = dreturn - value

        if self.use_scaled_ce:
            actor_ce_scaled = scaled_cross_entropy(
                labels= action_taken,
                scale=  advantage,
                probs=  out['probs'])
        else:
            actor_ce = torch.nn.functional.cross_entropy(logits, action_taken, reduction='none')
            actor_ce_scaled = actor_ce * advantage

        loss_actor_scaled_mean = torch.mean(actor_ce_scaled)

        if self.use_huber: loss_critic = torch.nn.functional.huber_loss(value, dreturn, reduction='none')
        else:              loss_critic = advantage^2  # MSE

        loss_critic_mean = torch.mean(loss_critic)

        out.update({
            'loss_actor':       loss_actor_scaled_mean,
            'loss_critic':      loss_critic_mean,
            'loss':             loss_actor_scaled_mean + loss_critic_mean,
            'acc':              self.accuracy(logits, action_taken)})

        return out