import numpy as np
from typing import Optional, Callable

from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.policy_gradients.actor_critic.tf_based.ac_TF_critic_graph import critic_graph
from pypaq.R4C.helpers import RLException
from pypaq.neuralmess.nemodel import NEModel


class AC_TFCritic(PGActor):

    def __init__(
            self,
            name: str=                      'AC_TFCritic',
            nngraph: Optional[Callable]=    critic_graph,
            **kwargs):
        PGActor.__init__(
            self,
            name=       name,
            nnwrap=     NEModel,
            nngraph=    nngraph,
            **kwargs)

    # Critic does not have policy
    def get_policy_probs(self, observation: object) -> np.ndarray:
        raise RLException('not implemented since should not be called')

    def get_qvs(self, observation) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        out = self.nnw(
            feed_dict=  {self.nnw['observation_PH']: [obs_vec]},
            fetch=      ['qvs'])
        return out['qvs']

    def get_qvs_batch(self, observations) -> np.ndarray:
        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.nnw(
            feed_dict=  {self.nnw['observation_PH']: obs_vecs},
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
        out = self.nnw.backward(
            feed_dict=  {
                self.nnw['observation_PH']:       obs_vecs,
                self.nnw['action_OH_PH']:         actions_OH,
                self.nnw['next_action_qvs_PH']:   next_action_qvs,
                self.nnw['next_action_probs_PH']: next_actions_probs,
                self.nnw['reward_PH']:            rewards},
            fetch=      ['optimizer','loss','gg_norm','gg_avt_norm'])
        out.pop('optimizer')
        return out