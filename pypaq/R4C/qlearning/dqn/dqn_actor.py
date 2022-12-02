"""

 2022 (c) piteren

    DQN_Actor - Deep QLearningActor NN based

"""

from abc import ABC
from typing import Optional, Union, Callable

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.little_methods import stamp
from pypaq.R4C.qlearning.ql_actor import QLearningActor
from pypaq.comoneural.nnwrap import NNWrap


# DQN (NN based) QLearningActor
class DQN_Actor(QLearningActor, ABC):

    def __init__(
            self,
            nnwrap: type(NNWrap),
            nngraph: Optional[Union[Callable, type]]=   None,
            name: Optional[str]=                        None,
            **kwargs):

        qla_kwargs = {k: kwargs[k] for k in ['envy','seed', 'logger', 'loglevel'] if k in kwargs}
        QLearningActor.__init__(self, **qla_kwargs)

        kwargs['num_actions'] = self._envy.num_actions()
        kwargs['observation_width'] = self._get_observation_vec(self._envy.get_observation()).shape[-1]
        if 'logger' in kwargs: kwargs.pop('logger')  # INFO: NNWrap will always create own loger with given level
        self.nnw: NNWrap = nnwrap(
            nngraph=    nngraph,
            name=       name or f'nn_{self.__class__.__name__}_{stamp()}',
            **kwargs)

        self._upd_step = 0

        self._log.info('*** DQN_Actor *** (NN based) initialized')
        self._log.info(f'> NNW model name:    {self.nnw["name"]}')
        self._log.info(f'> num_actions:       {self.nnw["num_actions"]}')
        self._log.info(f'> observation_width: {self.nnw["observation_width"]}')
        self._log.info(f'> save_topdir:       {self.nnw["save_topdir"]}')

    # INFO: wont be used since DQN_Actor updates only with batches
    def _upd_QV(
            self,
            observation: object,
            action: int,
            new_qv: float) -> float:
        raise Exception('not implemented')

    def save(self) -> None: self.nnw.save()

    def __str__(self) -> str: return str(self.nnw)