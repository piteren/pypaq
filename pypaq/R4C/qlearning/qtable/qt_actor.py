"""

 2022 (c) piteren

    QTableActor: QLearningActor with QTable

    QTableActor may be trained by QLearningTrainer. Trainer is responsible for computation of new QV for an Actor.
    Actor on the other side has its own QV update_rate, which works as a kind of Actor-specific learning ratio.
    Learning process is mainly directed by Trainer with possible little override by an Actor.
    Actor is responsible for computation of its loss.

"""

import numpy as np
from typing import Hashable, Dict, List

from pypaq.R4C.qlearning.ql_actor import QLearningActor
from pypaq.R4C.helpers import RLException


class QTable:

    def __init__(self, width: int):
        self.__width = width
        self.__table: Dict[Hashable, np.ndarray] = {}  # {observation_hash: QVs}
        self.__keys: List[np.ndarray] = []

    @staticmethod
    def __hash(observation: np.ndarray) -> str:
        return str(observation)

    def __init_hash(self, ha: str):
        self.__table[ha] = np.zeros(self.__width, dtype=float)

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
class QTableActor(QLearningActor):

    def __init__(
            self,
            name: str=          'QTableActor',
            save_topdir: str=   '_models',
            **kwargs):

        QLearningActor.__init__(self, name=name, **kwargs)

        self._save_topdir = save_topdir
        self.__qtable = QTable(self._envy.num_actions())
        self._update_rate = None # needs to be set before update

    def set_update_rate(self, update_rate:float):
        self._update_rate = update_rate
        self._rlog.info(f'> QTableActor set update_rate to: {self._update_rate}')

    def _get_QVs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        return self.__qtable.get_QVs(obs_vec)

    def _upd_QV(
            self,
            observation: object,
            action: int,
            new_qv: float) -> float:
        if self._update_rate is None: raise RLException('Trainer needs to set update_rate of QLearningActor before training!')
        old_qv = self._get_QVs(observation)[action]
        diff = new_qv - old_qv
        self.__qtable.put_QV(
            observation=    self._get_observation_vec(observation),
            action=         action,
            new_qv=         old_qv + self._update_rate * diff)
        return abs(diff)

    def _get_save_topdir(self) -> str:
        return self._save_topdir

    def save(self):
        raise RLException('not implemented')

    def __str__(self):
        return f'QTableActor, QTable:\n{self.__qtable.__str__()}'