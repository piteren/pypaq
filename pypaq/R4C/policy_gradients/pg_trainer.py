"""

 2022 (c) piteren

    Policy Gradients Trainer

"""

from pypaq.lipytools.plots import two_dim_multi
from pypaq.R4C.helpers import zscore_norm, extract_from_batch
from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.trainer import FATrainer


class PGTrainer(FATrainer):

    def __init__(
            self,
            actor: PGActor,
            **kwargs):

        FATrainer.__init__(
            self,
            actor=  actor,
            **kwargs)
        self.actor = actor # INFO: type "upgrade" for pycharm editor

    # PGActor update method
    def update_actor(
            self,
            reset_memory=   True,
            inspect=        False) -> float:

        batch = self.memory.sample(self.batch_size)

        observations = extract_from_batch(batch, 'observation')
        actions =      extract_from_batch(batch, 'action')
        dreturns =     extract_from_batch(batch, 'dreturn')
        rewards =      extract_from_batch(batch, 'reward')

        dreturns_norm = zscore_norm(dreturns)

        if inspect:
            two_dim_multi(
                ys=         [rewards, dreturns, dreturns_norm],
                names=      ['rewards', 'dreturns', 'dreturns_norm'],
                legend_loc= 'lower left')

        loss = self.actor.update_with_experience(
            observations=   observations,
            actions=        actions,
            dreturns=       dreturns_norm)

        if reset_memory: self.memory.reset()

        return loss