import numpy as np
from typing import Optional, List, Dict, Any

from pypaq.torchness.motorch import Module
from pypaq.R4C.helpers import extract_from_batch
from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.policy_gradients.actor_critic.ac_critic_module import ACCriticModule
from pypaq.R4C.helpers import RLException


class ACCritic(PGActor):

    def __init__(
            self,
            name: str=                              'ACCritic',
            module_type: Optional[type(Module)]=    ACCriticModule,
            **kwargs):
        PGActor.__init__(
            self,
            name=           name,
            module_type=    module_type,
            **kwargs)

    # Critic does not have policy
    def get_policy_probs(self, observation: object) -> np.ndarray:
        raise RLException('not implemented since should not be called')

    # TODO: what about shape, does it work?
    def get_qvs(self, observation) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        out = self.model(obs_vec)
        return out['qvs'].detach().cpu().numpy()


    def get_qvs_batch(self, observations) -> np.ndarray:
        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.model(obs_vecs)
        return out['qvs'].detach().cpu().numpy()


    def update_with_experience(
            self,
            batch: List[Dict[str, Any]],
            inspect: bool
    ) -> Dict[str, Any]:

        observations = extract_from_batch(batch, 'observation')
        obs_vecs = self._get_observation_vec_batch(observations)

        out = self.model.backward(
            observation=        obs_vecs,
            action_taken_OH=    extract_from_batch(batch, 'action_OH'),
            next_action_qvs=    extract_from_batch(batch, 'next_action_qvs'),
            next_action_probs=  extract_from_batch(batch, 'next_action_probs'),
            reward=             extract_from_batch(batch, 'reward'))
        out.pop('qvs')
        return out