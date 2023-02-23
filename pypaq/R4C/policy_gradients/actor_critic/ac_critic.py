import numpy as np
from typing import Optional

from pypaq.torchness.motorch import Module
from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.policy_gradients.actor_critic.ac_critic_module import ACModule
from pypaq.R4C.helpers import RLException


class ACCritic(PGActor):

    def __init__(
            self,
            name: str=                              'ACCritic',
            module_type: Optional[type(Module)]=    ACModule,
            **kwargs):
        PGActor.__init__(
            self,
            name=           name,
            module_type=    module_type,
            **kwargs)

    # Critic does not have policy
    def get_policy_probs(self, observation: object) -> np.ndarray:
        raise RLException('not implemented since should not be called')

    # TODO: implement for Module
    def get_qvs(self, observation) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        out = self.model(
            feed_dict=  {self.model['observation_PH']: [obs_vec]},
            fetch=      ['qvs'])
        return out['qvs']

    def get_qvs_batch(self, observations) -> np.ndarray:
        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.model(
            feed_dict=  {self.model['observation_PH']: obs_vecs},
            fetch=      ['qvs'])
        return out['qvs']

    def update_with_experience(
            self,
            observations,
            actions_OH,
            next_action_qvs,
            next_actions_probs,
            rewards,
            inspect=    False) -> dict:
        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.model.backward(
            feed_dict=  {
                self.model['observation_PH']:       obs_vecs,
                self.model['action_OH_PH']:         actions_OH,
                self.model['next_action_qvs_PH']:   next_action_qvs,
                self.model['next_action_probs_PH']: next_actions_probs,
                self.model['reward_PH']:            rewards},
            fetch=      ['optimizer','loss','gg_norm','gg_avt_norm'])
        out.pop('optimizer')
        return out