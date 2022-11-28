"""

 2022 (c) piteren

    DQN_Actor - Actor with NN, common interface

"""

from abc import ABC
from typing import Optional, Union, Callable

from pypaq.lipytools.little_methods import stamp
from pypaq.R4C.qlearning.qlearning_actor import QLearningActor
from pypaq.R4C.envy import FiniteActionsRLEnvy
from pypaq.comoneural.nnwrap import NNWrap


# DQN (NN based) QLearningActor
class DQN_Actor(QLearningActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            nnwrap: type(NNWrap),
            seed: int,
            logger,
            nngraph: Optional[Union[Callable, type]]=   None,
            name: Optional[str]=                        None,
            **kwargs):

        self.__log = logger

        QLearningActor.__init__(
            self,
            envy=   envy,
            seed=   seed,
            logger= self.__log)
        self._envy = envy # to update type (for pycharm only)

        if not name not in kwargs: name = f'nn_{self.__class__.__name__}_{stamp()}'
        kwargs['num_actions'] = self._envy.num_actions()
        kwargs['observation_width'] = self._get_observation_vec(self._envy.get_observation()).shape[-1]

        self.nnw: NNWrap = nnwrap(
            nngraph=    nngraph,
            name=       name,
            **kwargs)

        self._upd_step = 0

        self.__log.info('*** DQN_Actor *** (NN based) initialized')
        self.__log.info(f'> NNW model name:    {self.nnw["name"]}')
        self.__log.info(f'> num_actions:       {self.nnw["num_actions"]}')
        self.__log.info(f'> observation_width: {self.nnw["observation_width"]}')
        self.__log.info(f'> save_topdir:       {self.nnw["save_topdir"]}')

    # INFO: wont be used since DQN_Actor updates only with batches
    def _upd_QV(
            self,
            observation: object,
            action: int,
            new_qv: float) -> float:
        raise Exception('not implemented')

    def save(self) -> None: self.nnw.save()

    def __str__(self) -> str: return str(self.nnw)