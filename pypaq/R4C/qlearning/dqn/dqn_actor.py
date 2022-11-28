"""

 2022 (c) piteren

    DQN_Actor - Actor with NN, common interface

"""

from abc import ABC

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
            # mdict: dict, # TODO: give it with kwargs
            **kwargs):

        self._envy = envy

        if 'name' not in kwargs: kwargs['name'] = f'nn_{self.__class__.__name__}_{stamp()}'
        kwargs['num_actions'] = self._envy.num_actions()
        kwargs['observation_width'] = self._get_observation_vec(self._envy.get_observation()).shape[-1]

        nnwrap.__init__(self, **kwargs)

        self._upd_step = 0

        self._log.info('*** DQN_Actor (NN based) initialized')
        self._log.info(f'> Envy:              {self._envy.__class__.__name__}')
        self._log.info(f'> NN model name:     {self["name"]}')
        self._log.info(f'> num_actions:       {self["num_actions"]}')
        self._log.info(f'> observation_width: {self["observation_width"]}')
        self._log.info(f'> save_topdir:       {self["save_topdir"]}')

    # INFO: not used since DQN_Actor updates only with batches
    def _upd_QV(
            self,
            observation: object,
            action: int,
            new_qv: float) -> float:
        raise Exception('not implemented')