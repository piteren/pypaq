"""

 2022 (c) piteren

    DQNActor - QLearningActor based on NN

    DQNActor updates NN weights (self state) using optimizer & loss calculated with given gold QV.

"""

from abc import abstractmethod, ABC
import numpy as np

from pypaq.lipytools.little_methods import stamp
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.qlearning.qlearning_actor import QLearningActor


# QLearningActor with NeuralNetwork (DQN)
class DQNActor(QLearningActor, ABC):

    def __init__(
            self,
            num_actions: int,
            observation,
            mdict: dict,
            logger,
            name_pfx=   'dqn'):

        self.__log = logger

        if 'name' not in mdict: mdict['name'] = f'{name_pfx}_{stamp()}'
        mdict['num_actions'] = num_actions
        mdict['observation_width'] = self.get_observation_vec(observation).shape[-1]

        self.nn = None

        self.__log.info(f'*** DQNActor {mdict["name"]} initialized, num_actions: {mdict["num_actions"]}, observation_width: {mdict["observation_width"]}')

    # initializes self.nn
    @abstractmethod
    def _init_model(self, **kwargs): pass

    # INFO: not used since DQNActor updates only with batches
    def upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float:
        raise NotImplementedError

    def save(self): self.nn.save()

    def __str__(self): return self.nn.__str__()