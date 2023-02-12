"""

 2022 (c) piteren

    DQN_Actor - Deep QLearningActor NN based

"""

from abc import ABC
import numpy as np
from typing import Optional, Union, Callable, List

from pypaq.R4C.qlearning.ql_actor import QLearningActor
from pypaq.R4C.helpers import RLException
from pypaq.torchness.comoneural import NNWrap


# DQN (NN based) QLearningActor
class DQN_Actor(QLearningActor, ABC):

    def __init__(
            self,
            nnwrap: type(NNWrap),
            module_type: Optional[Union[Callable, type]]=   None,
            **kwargs):

        QLearningActor.__init__(self, **kwargs)

        # some overrides and updates
        kwargs['name'] = self.name
        kwargs['name_timestamp'] = False                # name timestamp is driven by TrainableActor (with self.name)
        if 'logger' in kwargs: kwargs.pop('logger')     # NNWrap will always create own logger (since then it is not given) with optionally given level
        kwargs['num_actions'] = self._envy.num_actions()
        kwargs['observation_width'] = self._get_observation_vec(self._envy.get_observation()).shape[-1]

        self.nnw: NNWrap = nnwrap(module_type=module_type, **kwargs)

        self._rlog.info('*** DQN_Actor *** initialized')
        self._rlog.info(f'> NNWrap: {nnwrap.__name__}')

    # vectorization of observations batch, may be overridden with more optimal custom implementation
    def _get_observation_vec_batch(self, observations: List[object]) -> np.ndarray:
        return np.asarray([self._get_observation_vec(v) for v in observations])

    # INFO: wont be used since DQN_Actor updates only with batches
    def _upd_QV(
            self,
            observation: object,
            action: int,
            new_qv: float) -> float:
        raise RLException('not implemented')

    def _get_save_topdir(self) -> str:
        return self.nnw['save_topdir']

    def save(self):
        self.nnw.save()

    def __str__(self) -> str:
        return str(self.nnw)