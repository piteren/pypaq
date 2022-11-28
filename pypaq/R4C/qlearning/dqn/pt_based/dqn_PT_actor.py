"""

 2022 (c) piteren

    DQNPTActor - DQNActor implemented with MOTorch (PyTorch)

"""

from abc import ABC
import numpy as np
from typing import List, Optional

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.qlearning.dqn.dqn_actor import DQN_Actor
from pypaq.R4C.qlearning.dqn.pt_based.dqn_PT_module import LinModel
from pypaq.torchness.motorch import MOTorch, Module


# DQN (PyTorch NN) based QLearningActor
class DQN_PTActor(DQN_Actor, ABC):

    def __init__(
            self,
            nngraph: Optional[type(Module)]=    LinModel,
            logger=                             None,
            loglevel=                           20,
            **kwargs):

        if not logger:
            logger = get_pylogger(
                name=       'DQN_PTActor',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.__log = logger

        DQN_Actor.__init__(
            self,
            nnwrap=     MOTorch,
            nngraph=    nngraph,
            logger=     self.__log,
            **kwargs)

    def _get_QVs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        return self.nnw(obs_vec)['logits'].detach().cpu().numpy()

    # optimized with single call with a batch of observations
    def get_QVs_batch(self, observations: List[object]) -> np.ndarray:
        obs_vecs = np.array([self._get_observation_vec(o) for o in observations])
        return self.nnw(obs_vecs)['logits'].detach().cpu().numpy()

    # optimized with single call to session with a batch of data
    def update_with_experience(
            self,
            observations: List[object],
            actions: List[int],
            new_qvs: List[float],
            inspect=    False) -> float:

        obs_vecs = np.array([self._get_observation_vec(o) for o in observations])
        full_qvs = np.zeros_like(obs_vecs)
        mask = np.zeros_like(obs_vecs)
        for v,pos in zip(new_qvs, enumerate(actions)):
            full_qvs[pos] = v
            mask[pos] = 1

        self.__log.log(5, f'>>> obs_vecs.shape, len(actions), new_qvs.shape: {obs_vecs.shape}, {len(actions)}, {len(new_qvs)}')
        self.__log.log(5, f'>>> actions: {actions}')
        self.__log.log(5, f'>>> new_qvs: {new_qvs}')
        self.__log.log(5, f'>>> full_qvs: {full_qvs}')
        self.__log.log(5, f'>>> mask: {mask}')

        out = self.nnw.backward(obs_vecs, full_qvs, mask)

        self._upd_step += 1

        loss = float(out['loss'])
        gn = float(out['gg_norm'])
        gn_avt = float(out['gg_avt_norm'])
        cLR = float(out['currentLR'])

        self.nnw.log_TB(loss,    'upd/loss',     step=self._upd_step)
        self.nnw.log_TB(gn,      'upd/gn',       step=self._upd_step)
        self.nnw.log_TB(gn_avt,  'upd/gn_avt',   step=self._upd_step)
        self.nnw.log_TB(cLR,     'upd/cLR',      step=self._upd_step)

        return loss