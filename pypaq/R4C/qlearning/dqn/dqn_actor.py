"""

 2022 (c) piteren

    DQN_Actor - Deep QLearningActor NN based

"""

from abc import ABC
import numpy as np
from typing import Optional, List

from pypaq.R4C.qlearning.ql_actor import QLearningActor
from pypaq.R4C.helpers import RLException
from pypaq.torchness.motorch import MOTorch, Module
from pypaq.R4C.qlearning.dqn.dqn_actor_module import DQNModel


# DQN (NN based) QLearningActor
class DQNActor(QLearningActor, ABC):

    def __init__(
            self,
            name: str=                              'DQNActor',
            module_type: Optional[type(Module)]=    DQNModel,
            seed: int=                              123,
            **kwargs):

        QLearningActor.__init__(
            self,
            name=   name,
            seed=   seed,
            **kwargs)

        # some overrides and updates
        if 'logger' in kwargs: kwargs.pop('logger')     # NNWrap will always create own logger (since then it is not given) with optionally given level
        kwargs['num_actions'] = self._envy.num_actions()
        kwargs['observation_width'] = self._get_observation_vec(self._envy.get_observation()).shape[-1]

        self.model = MOTorch(
            module_type=    module_type,
            name=           self.name,
            seed=           seed,
            **kwargs)

        self._rlog.info(f'*** DQNActor : {self.name} *** initialized')

    def _get_QVs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        return self.model(obs_vec)['logits'].detach().cpu().numpy()

    # optimized with single call with a batch of observations
    def get_QVs_batch(self, observations: List[object]) -> np.ndarray:
        obs_vecs = np.asarray([self._get_observation_vec(o) for o in observations])
        return self.model(obs_vecs)['logits'].detach().cpu().numpy()

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

    # optimized with single call to session with a batch of data
    def update_with_experience(
            self,
            observations: List[object],
            actions: List[int],
            new_qvs: List[float],
            inspect=    False) -> dict:

        obs_vecs = self._get_observation_vec_batch(observations)
        full_qvs = np.zeros_like(obs_vecs)
        mask = np.zeros_like(obs_vecs)
        for v,pos in zip(new_qvs, enumerate(actions)):
            full_qvs[pos] = v
            mask[pos] = 1

        self._rlog.log(5, f'>> obs_vecs (shape {obs_vecs.shape})\n{obs_vecs}')
        self._rlog.log(5, f'>> actions (len {len(actions)}): {actions}')
        self._rlog.log(5, f'>> new_qvs (len {len(new_qvs)}): {new_qvs}')
        self._rlog.log(5, f'>> full_qvs\n{full_qvs}')
        self._rlog.log(5, f'>> mask\n{mask}')

        out = self.model.backward(obs_vecs, full_qvs, mask)
        out.pop('logits')
        out.pop('acc') # accuracy for DQN does not make sense
        return out

    def _get_save_topdir(self) -> str:
        return self.model['save_topdir']

    def save(self):
        self.model.save()

    def __str__(self) -> str:
        return str(self.model)