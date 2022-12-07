"""

 2022 (c) piteren

    QLearningTrainer for QLearningActor acting on FiniteActionsRLEnvy.

"""

import numpy as np

from pypaq.R4C.helpers import extract_from_batch
from pypaq.R4C.trainer import FATrainer
from pypaq.R4C.qlearning.ql_actor import QLearningActor


# Q-Learning Trainer
class QLearningTrainer(FATrainer):

    def __init__(
            self,
            actor: QLearningActor,
            gamma: float,       # QLearning gamma (discount)
            **kwargs):

        FATrainer.__init__(self, actor=actor, **kwargs)
        self.actor = actor  # INFO: just type "upgrade" for pycharm editor
        self.gamma = gamma

        self._rlog.info(f'*** QLearningTrainer *** initialized')
        self._rlog.info(f'> gamma: {self.gamma}')

    # updates QLearningActor policy with batch of random data from memory
    def _update_actor(self, inspect=False) -> dict:

        batch = self.memory.sample(self.batch_size)

        observations =      extract_from_batch(batch, 'observation')
        actions =           extract_from_batch(batch, 'action')
        rewards =           extract_from_batch(batch, 'reward')
        next_observations = extract_from_batch(batch, 'next_observation')
        terminals =         extract_from_batch(batch, 'terminal')

        no_qvs = self.actor.get_QVs_batch(next_observations)
        no_qvs_terminal = np.zeros(self.envy.num_actions())

        for ix,t in enumerate(terminals):
            if t: no_qvs[ix] = no_qvs_terminal

        new_qvs = [(r + self.gamma * max(no_qvs)) for r, no_qvs in zip(rewards, no_qvs)]

        return self.actor.update_with_experience(
            observations=   observations,
            actions=        actions,
            new_qvs=        new_qvs,
            inspect=        inspect)