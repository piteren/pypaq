"""

 2022 (c) piteren

    QLearningActor with QTable

    QLearningActor may be trained by QLearningTrainer. Trainer is responsible by computation of new QV for an Actor.
    Actor on the other side has its own QV update rate (upd_rate), which works as a kind of Actor-specific learning ratio.
    Since then learning process is mainly directed by Trainer with possible little override by an Actor.

"""

from abc import ABC
import numpy as np
from typing import Hashable, Dict, List

from pypaq.R4C.qlearning.qlearning_actor import QLearningActor


class QTable:

    def __init__(self, width: int):
        self.__width = width
        self.__table: Dict[Hashable, np.ndarray] = {}  # {observation_hash: QVs}
        self.__keys: List[np.ndarray] = []

    @staticmethod
    def __hash(observation: np.ndarray) -> str:
        return str(observation)

    def __init_hash(self, ha: str):
        self.__table[ha] = np.zeros(self.__width, dtype=np.float)

    def get_QVs(self, observation: np.ndarray) -> np.ndarray:
        ha = QTable.__hash(observation)
        if ha not in self.__table:
            self.__init_hash(ha)
            self.__keys.append(observation)
        return self.__table[ha]

    def put_QV(self,
            observation: np.ndarray,
            action: int,
            new_qv: float):
        ha = QTable.__hash(observation)
        if ha not in self.__table:
            self.__init_hash(ha)
            self.__keys.append(observation)
        self.__table[ha][action] = new_qv

    def __str__(self):
        s = f'QTable:\n'
        if self.__table:
            for k in sorted(self.__table.keys()):
                s += f'{k} : {self.__table[k]}\n'
        else: s += '<empty>\n'
        return s


# implements QLearningActor with QTable
class QTableActor(QLearningActor, ABC):

    def __init__(
            self,
            num_actions: int,
            upd_rate=   0.5): # QV update rate
        self.__upd_rate = upd_rate
        self.__qtable = QTable(num_actions)

    def get_QVs(self, observation: np.ndarray) -> np.ndarray:
        return self.__qtable.get_QVs(observation)

    def upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float:
        old_qv = self.get_QVs(observation)[action]
        diff = new_qv - old_qv
        self.__qtable.put_QV(
            observation=    observation,
            action=         action,
            new_qv=         old_qv + self.__upd_rate * diff)
        return abs(diff)

    def __str__(self): return self.__qtable.__str__()