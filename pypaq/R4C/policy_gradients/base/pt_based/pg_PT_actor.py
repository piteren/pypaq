"""

 2022 (c) piteren

    PolicyGradients NN Actor, PT based

"""

from abc import ABC
import numpy as np
from typing import Optional

from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.policy_gradients.base.pt_based.pg_PT_module import PGModel
from pypaq.torchness.motorch import MOTorch, Module



class PG_PTActor(PGActor, ABC):

    def __init__(
            self,
            name: str=                              'PG_PTActor',
            module_type: Optional[type(Module)]=    PGModel,
            **kwargs):
        PGActor.__init__(
            self,
            name=           name,
            nnwrap=         MOTorch,
            module_type=    module_type,
            **kwargs)

    def get_policy_probs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        return self.nnw(obs_vec)['probs'].detach().cpu().numpy()

    # optimized with batch call to NN
    def get_policy_probs_batch(self, observations) -> np.ndarray:
        obs_vecs = self._get_observation_vec_batch(observations)
        return self.nnw(obs_vecs)['probs'].detach().cpu().numpy()

    # updates self NN with batch of data
    def update_with_experience(
            self,
            observations,
            actions,
            dreturns,
            inspect=    False) -> dict:
        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.nnw.backward(obs_vecs, actions, dreturns)
        out.pop('logits')
        if 'probs' in out: out['probs'] = out['probs'].cpu().detach().numpy()
        if 'zeroes' in out: out['zeroes'] = [e.cpu().detach().numpy() for e in out['zeroes']]
        return out