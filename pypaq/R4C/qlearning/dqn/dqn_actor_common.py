"""

 2022 (c) piteren

    DQNN - QLearningActor based on NN (NEModel)

    Similar to QLearningActor, DQNN has its own QV update override implemented with NN Optimizer that is responsible by
    updating NN weights with call of optimizer policies like learning ratio and other.
    DQNN updates not QV but NN weights according to loss calculated with given gold QV.

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
            logger=     None):

        if 'name' not in mdict: mdict['name'] = f'dqn_{stamp()}'

        if not logger:
            logger = get_pylogger(
                name=       mdict['name'],
                add_stamp=  False,
                folder=     None)
        self.__log = logger

        mdict['num_actions'] = num_actions
        mdict['observation_width'] = self.get_observation_vec(observation).shape[-1]

        self.nn = None

        self.__log.info(f'*** DQNActor ({mdict["name"]}) inits, num_actions: {num_actions}, observation_width: {mdict["observation_width"]}')

    @abstractmethod
    def _init_model(self, **kwargs): pass

    # INFO: not used since DQNActor updates only with batches
    def upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float:
        raise NotImplementedError

    def save(self):
        self.nn.save()