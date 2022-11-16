"""

 2022 (c) piteren

    QTableActor: QLearningActor with QTable

    QTableActor may be trained by QLearningTrainer. Trainer is responsible for computation of new QV for an Actor.
    Actor on the other side has its own QV update_rate, which works as a kind of Actor-specific learning ratio.
    Learning process is mainly directed by Trainer with possible little override by an Actor.
    Actor is responsible for computation of its loss.

"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Hashable, Dict, List

from pypaq.R4C.qlearning.qlearning_actor import QLearningActor
from pypaq.lipytools.pylogger import get_pylogger


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
        s = f'length: {len(self.__table) if self.__table else "<empty>"}\n'
        if self.__table:
            for k in sorted(self.__table.keys()):
                s += f'{k} : {self.__table[k]}\n'
        return s[:-1]


# implements QLearningActor with QTable
class QTableActor(QLearningActor, ABC):

    def __init__(
            self,
            num_actions: int,
            update_rate: float= 0.3,
            logger=             None,
            loglevel=           20):

        logger_given = bool(logger)
        if not logger_given:
            logger = get_pylogger(
                name=       'QTableActor',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.__log = logger
        self.__log.info(f'*** QTableActor initializes, num_actions: {num_actions}, update_rate: {update_rate}')

        self.__qtable = QTable(num_actions)
        self._update_rate = update_rate


    def set_update_rate(self, update_rate:float):
        self._update_rate = update_rate
        self.__log.info(f'> QTableActor set update_rate to: {self._update_rate}')

    # prepares numpy vector from observation, it is a private / internal skill of Actor
    @abstractmethod
    def _get_observation_vec(self, observation: object) -> np.ndarray: pass

    def _get_QVs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        return self.__qtable.get_QVs(obs_vec)

    def upd_QV(
            self,
            observation: object,
            action: int,
            new_qv: float) -> float:
        old_qv = self._get_QVs(observation)[action]
        diff = new_qv - old_qv
        self.__qtable.put_QV(
            observation=    self._get_observation_vec(observation),
            action=         action,
            new_qv=         old_qv + self._update_rate * diff)
        return abs(diff)

    def __str__(self):
        return f'QTableActor, QTable:\n{self.__qtable.__str__()}'