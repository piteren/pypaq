"""

 2022 (c) piteren

    DQNPTActor - DQNActor implemented with MOTorch (PyTorch)

"""

from abc import ABC
import numpy as np
from typing import List

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.qlearning.dqn.dqn_actor import DQN_Actor
from pypaq.R4C.qlearning.dqn.pt_based.dqn_PT_module import LinModel
from pypaq.torchness.motorch import MOTorch


# DQN (PyTorch NN) based QLearningActor
class DQN_PTActor(DQN_Actor, ABC):

    def __init__(
            self,
            mdict: dict,
            module=         LinModel,
            save_topdir=    '_models', # just to set default
            logger=         None,
            loglevel=       20,
            **kwargs):

        self._logger_given = bool(logger)
        self._loglevel = loglevel
        if not self._logger_given:
            logger = get_pylogger(
                name=       self.__class__.__name__,
                add_stamp=  True,
                folder=     save_topdir,
                level=      self._loglevel)
        self.__log = logger
        self.__log.info('*** DQN_PTActor (PyTorch based) initializes..')
        self.__log.info(f'> Module: {module.__name__}')

        self._module = module
        self._mdict = mdict

        DQN_Actor.__init__(
            self,
            mdict=          self._mdict,
            save_topdir=    save_topdir,
            logger=         self.__log,
            **kwargs)

        self.__log.info(f'DQN_PTActor initialized')

    def _get_model(self):
        return MOTorch(
            module=     self._module,
            logger=     None if not self._logger_given else self.__log,
            loglevel=   self._loglevel,
            **self._mdict)

    def _get_QVs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        return self.nn(obs_vec)['logits'].detach().cpu().numpy()

    # optimized with single call with a batch of observations
    def get_QVs_batch(self, observations: List[object]) -> np.ndarray:
        obs_vecs = np.array([self._get_observation_vec(o) for o in observations])
        return self.nn(obs_vecs)['logits'].detach().cpu().numpy()

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

        out = self.nn.backward(obs_vecs, full_qvs, mask)

        self._upd_step += 1

        loss = float(out['loss'])
        gn = float(out['gg_norm'])
        gn_avt = float(out['gg_avt_norm'])
        cLR = float(out['currentLR'])

        self.nn.log_TB(loss,    'upd/loss',     step=self._upd_step)
        self.nn.log_TB(gn,      'upd/gn',       step=self._upd_step)
        self.nn.log_TB(gn_avt,  'upd/gn_avt',   step=self._upd_step)
        self.nn.log_TB(cLR,     'upd/cLR',      step=self._upd_step)

        return loss

    def save(self): self.nn.save()

    def __str__(self): return self.nn.__str__()