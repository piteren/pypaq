"""

 2022 (c) piteren

    Policy Gradients Trainer

    PGTrainer implements sampled policy action - needs some methods to be rewritten
    to support "sampled" argument, that is set to True while training.

"""

from pypaq.lipytools.plots import two_dim_multi
from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.trainer import FATrainer


class PGTrainer(FATrainer):

    def __init__(self, actor: PGActor, **kwargs):

        FATrainer.__init__(self, actor=actor, **kwargs)
        self.actor = actor # INFO: type "upgrade" for pycharm editor

    # PGActor update method
    def update_actor(self, inspect=False) -> float:
        batch = self.memory.get_all()

        dreturns = self._extract_from_batch(batch, 'dreturn')
        dreturns_norm = self.zscore_norm(dreturns)

        if inspect:
            rewards = self._extract_from_batch(batch, 'reward')
            two_dim_multi(
                ys=         [rewards, dreturns, dreturns_norm],
                names=      ['rewards', 'dreturns', 'dreturns_norm'],
                legend_loc= 'lower left')

        loss = self.actor.update_batch(
            observations=   self._extract_from_batch(batch, 'observation'),
            actions=        self._extract_from_batch(batch, 'action'),
            dreturns=       dreturns_norm)
        self.memory.reset()
        return loss