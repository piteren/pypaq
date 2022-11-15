"""

 2022 (c) piteren

    QLearningTrainer for QLearningActor acting on FiniteActionsRLEnvy.

"""

import numpy as np
import random

from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.R4C.helpers import extract_from_batch
from pypaq.R4C.envy import FiniteActionsRLEnvy
from pypaq.R4C.trainer import FATrainer
from pypaq.R4C.qlearning.qlearning_actor import QLearningActor


# Q-Learning Trainer (used by QTable and DQN)
class QLearningTrainer(FATrainer):

    def __init__(
            self,
            actor: QLearningActor,
            envy: FiniteActionsRLEnvy,
            batch_size=         10,
            memsize_batches=    10,   # ExperienceMemory size (in number of batches)
            exploration=        0.5,  # exploration factor
            discount=           0.9,  # discount factor (gamma)
            seed=               123,
            logger=             None,
            loglevel=           20):

        if not logger:
            logger = get_pylogger(
                name=       'QLearningTrainer',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.__log = logger

        random.seed(seed)

        FATrainer.__init__(
            self,
            actor=              actor,
            envy=               envy,
            batch_size=         batch_size,
            memsize_batches=    memsize_batches,
            exploration=        exploration,
            discount=           discount,
            logger=             get_hi_child(self.__log, 'FATrainer', higher_level=False))
        self.actor = actor # INFO: just type "upgrade" for pycharm editor

        self.__log.info(f'*** QLearningTrainer for {envy.name} (actions: {self.num_of_actions})')

    # updates QLearningActor policy with batch of data from memory
    def update_actor(
            self,
            reset_memory=   True,
            inspect=        False) -> float: # INFO: inspect is not used ..

        batch = self.memory.sample(self.batch_size)

        observations =      extract_from_batch(batch, 'observation')
        actions =           extract_from_batch(batch, 'action')
        rewards =           extract_from_batch(batch, 'reward')
        next_observations = extract_from_batch(batch, 'next_observation')
        terminals =         extract_from_batch(batch, 'terminal')

        no_qvs = self.actor.get_QVs_batch(next_observations)
        no_qvs_terminal = np.zeros(self.num_of_actions)

        for ix,t in enumerate(terminals):
            if t: no_qvs[ix] = no_qvs_terminal

        new_qvs = np.array([(r + self.discount * max(no_qvs)) for r,no_qvs in zip(rewards, no_qvs)])

        if reset_memory: self.memory.reset()

        return self.actor.update_with_experience(
            observations=   observations,
            actions=        actions,
            new_qvs=        new_qvs)