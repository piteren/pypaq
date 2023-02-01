"""

 2022 (c) piteren

    DQNPTActor - DQNActor implemented with MOTorch (PyTorch)

"""

import numpy as np
from typing import List, Optional

from pypaq.R4C.qlearning.dqn.dqn_actor import DQN_Actor
from pypaq.R4C.qlearning.dqn.pt_based.dqn_PT_module import DQNModel
from pypaq.torchness.motorch import MOTorch, Module


# DQN (PyTorch NN) based QLearningActor
class DQN_PTActor(DQN_Actor):

    def __init__(
            self,
            name: str=                              'DQN_PTActor',
            module_type: Optional[type(Module)]=    DQNModel,
            **kwargs):
        DQN_Actor.__init__(
            self,
            name=           name,
            nnwrap=         MOTorch,
            module_type=    module_type,
            **kwargs)

    def _get_QVs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        return self.nnw(obs_vec)['logits'].detach().cpu().numpy()

    # optimized with single call with a batch of observations
    def get_QVs_batch(self, observations: List[object]) -> np.ndarray:
        obs_vecs = np.asarray([self._get_observation_vec(o) for o in observations])
        return self.nnw(obs_vecs)['logits'].detach().cpu().numpy()

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

        out = self.nnw.backward(obs_vecs, full_qvs, mask)
        out.pop('logits')
        out.pop('acc') # accuracy for DQN does not make sense
        return out
