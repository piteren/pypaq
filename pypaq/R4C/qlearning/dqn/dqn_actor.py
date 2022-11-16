"""

 2022 (c) piteren

    DQN_Actor - Actor with NN, common interface

"""

from abc import ABC, abstractmethod

from pypaq.lipytools.little_methods import stamp
from pypaq.R4C.qlearning.qlearning_actor import QLearningActor
from pypaq.R4C.envy import FiniteActionsRLEnvy


# DQN (NN based) QLearningActor
class DQN_Actor(QLearningActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            mdict: dict,
            save_topdir,
            logger):

        self.__log = logger

        if 'name' not in mdict: mdict['name'] = f'nn_{self.__class__.__name__}_{stamp()}'
        mdict['num_actions'] = envy.num_actions()
        mdict['observation_width'] = self._get_observation_vec(envy.get_observation()).shape[-1]
        mdict['save_topdir'] = save_topdir

        self.__log.info('*** DQN_Actor (NN based) initializes..')
        self.__log.info(f'> Envy:              {envy.__class__.__name__}')
        self.__log.info(f'> NN model name:     {mdict["name"]}')
        self.__log.info(f'> num_actions:       {mdict["num_actions"]}')
        self.__log.info(f'> observation_width: {mdict["observation_width"]}')
        self.__log.info(f'> save_topdir:       {mdict["save_topdir"]}')

        self.nn = self._get_model()

        self._upd_step = 0

        self.__log.info(f'DQN_Actor initialized')

    @abstractmethod
    def _get_model(self) -> object: pass

    # INFO: not used since DQN_Actor updates only with batches
    def upd_QV(
            self,
            observation: object,
            action: int,
            new_qv: float) -> float:
        raise Exception('not implemented')