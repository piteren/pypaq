"""

 2022 (c) piteren

    QLearningActor with QTable

    QLearningActor may be trained by QLearningTrainer. Trainer is responsible by computation of new QV for an Actor.
    Actor on the other side has its own QV update rate (upd_rate),
    which acts as a kind of learning ratio (Actor-specific).
    Since then learning process is mainly directed by Trainer with possible little override by an Actor.

"""

from abc import abstractmethod, ABC
import numpy as np
from typing import Hashable, Dict

from pypaq.R4C.qlearning.qlearning_actor import QLearningActor


# implements QLearningActor with QTable
class QTableActor(QLearningActor, ABC):

    def __init__(
            self,
            num_actions: int,
            upd_rate=       0.5,  # QV update rate
    ):
        self.__num_actions = num_actions
        self.__upd_rate = upd_rate
        self.__table: Dict[Hashable,np.ndarray] = {} # {observation_hash: np.array(QValues)}

    # returns hashable version of observation
    @abstractmethod
    def observation_hash(self, observation) -> Hashable: pass

    # inits observation QVs with zeroes (optimal for terminal states)
    def __init_observation(self, observation):
        observation = self.observation_hash(observation)
        self.__table[observation] = np.zeros(self.__num_actions, dtype=np.float)

    def get_QVs(self, observation) -> np.ndarray:
        observation = self.observation_hash(observation)
        if observation not in self.__table: self.__init_observation(observation)
        return self.__table[observation]

    def upd_QV(
            self,
            observation,
            action: int,
            new_qv: float):
        observation = self.observation_hash(observation)
        if observation not in self.__table: self.__init_observation(observation)
        self.__table[observation][action] += self.__upd_rate * (new_qv - self.__table[observation][action])

    def __str__(self):
        s = f'Actor QTable:'
        keys = sorted(self.__table.keys())
        if keys:
            s += '\n'
            for obs in keys:
                s += f'{obs} : {self.__table[obs]}\n'
        else: s += ' <empty QTable>\n'
        return s