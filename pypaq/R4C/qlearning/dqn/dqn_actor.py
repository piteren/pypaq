"""

 2022 (c) piteren

    DQN_Actor - Actor with NN, common interface

"""

from abc import ABC, abstractmethod

from pypaq.lipytools.pylogger import get_pylogger
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
            logger=     None,
            loglevel=   20):

        self._logger_given = bool(logger)
        self._loglevel = loglevel
        if not self._logger_given:
            logger = get_pylogger(
                name=       self.__class__.__name__,
                add_stamp=  True,
                folder=     save_topdir,
                level=      self._loglevel)
        self._log = logger

        self._mdict = mdict
        if 'name' not in self._mdict: self._mdict['name'] = f'nn_{self.__class__.__name__}_{stamp()}'
        self._mdict['num_actions'] = envy.num_actions()
        self._mdict['observation_width'] = self._get_observation_vec(envy.get_observation()).shape[-1]
        self._mdict['save_topdir'] = save_topdir
        

        self._log.info('*** DQN_Actor (NN based) initializes..')
        self._log.info(f'> Envy:              {envy.__class__.__name__}')
        self._log.info(f'> NN model name:     {self._mdict["name"]}')
        self._log.info(f'> num_actions:       {self._mdict["num_actions"]}')
        self._log.info(f'> observation_width: {self._mdict["observation_width"]}')
        self._log.info(f'> save_topdir:       {self._mdict["save_topdir"]}')

        self.nn = self._get_model()

        self._upd_step = 0

        self._log.info(f'DQN_Actor initialized')

    @abstractmethod
    def _get_model(self) -> object: pass

    # INFO: not used since DQN_Actor updates only with batches
    def upd_QV(
            self,
            observation: object,
            action: int,
            new_qv: float) -> float:
        raise Exception('not implemented')